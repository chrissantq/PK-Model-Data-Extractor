from openai import AzureOpenAI
import xmlparser as xp
import pandas as pd
import os, io, re, csv, tiktoken


class FetchModelInformation:

  '''
    Class to define functionality to contextualize
    and fetch model data from both the freetext
    and extracted tables from each paper
  '''

  def __init__(self, runpath, llm_client, deployment):
    self.runpath = runpath
    self.llm_client = llm_client
    self.deployment = deployment

  # function to prime the llm for model information extraction
  def prime_llm(self, tags):

    # get title and freetext from article
    paper = xp.fetch_tags(self.runpath, tags)
    if paper is None:
      print(f"[ERROR] Failed to parse XML file at {self.runpath} - skipping")
      return -1
    xmlnodes = paper.text_node_list

    # build title + freetext to serve to AI model
    title = "The title of the paper is "
    freetext_list = ["This is the freetext of the article, you will answer questions about this article:"]
    for node in xmlnodes:
      if node.tag == "article-meta/title-group":
        title += node.text
      else:
        freetext_list.append(node.text)
    freetext = "\n".join(freetext_list)

    # remove disallowed special tokens
    freetext = re.sub(r"<\|.*?\|>", "", freetext)

    enc = tiktoken.encoding_for_model(self.deployment)
    MAX_TOK = 100000

    freetext_tokens = enc.encode(freetext)
    if len(freetext_tokens) > MAX_TOK:
      freetext_tokens = freetext_tokens[:MAX_TOK]
      freetext = enc.decode(freetext_tokens)

    # serve llm with the contextual information about the study
    try:
      response = self.llm_client.chat.completions.create(
        model=self.deployment,
        messages=[
          {"role": "system", "content": "You are an assistant that extracts information about the pharmacokinetics model used in a research paper."},
          {"role": "system", "content": "Ensure every cell is enclosed in double quotes, each row uses pipe delimiters, and every string is properly closed."},
          {"role": "user", "content": title},
          {"role": "user", "content": freetext},
        ]
      )
    except Exception as e:
      print(f"[ERROR] Error priming LLM")
      print(f"[ERROR] Exception: {e}")
      return -1

    return 0

  def clean_output(self, raw, pmcid):

    if not isinstance(raw, str):
      print(f"[{pmcid}] clean_output received non-string input: {type(raw)}")
      return pd.DataFrame(columns=["Data", "Value", "Relevance"])

    clean = re.sub(r"```(?:csv)?", "", raw, flags=re.I).strip()

    if not clean or "|" not in clean:
      print(f"[{pmcid}] Empty or invalid table")
      return pd.DataFrame(columns=["Data", "Value", "Relevance"])

    try:
      return pd.read_csv(
        io.StringIO(clean),
        sep="|",
        quoting=csv.QUOTE_MINIMAL,
        engine="c",
        dtype=str
      )
    except pd.errors.ParserError as e:
      print(f"ParserError in paper {pmcid}: {e}, retrying with python engine")
    except Exception as e:
      print(f"Unexpected error with C engine: {e}. Retrying with python engine")

    try:
      return pd.read_csv(
        io.StringIO(clean),
        sep="|",
        quoting=csv.QUOTE_MINIMAL,
        engine="python",
        on_bad_lines="skip",
        dtype=str
      )
    except pd.errors.EmptyDataError as e:
      print(f"Python engine failed to parse: {e}")
      return pd.DataFrame(columns=["Data", "Value", "Relevance"])
    except Exception as e:
      print(f"Unexpected error with Python engine: {e}")
      return pd.DataFrame(columns=["Data", "Value", "Relevance"])
    else:
      if df.shape[1] != 3:
        print(f"[{pmcid}] Warning: parsed DataFrame has {df.shape[1]} columns instead of 3")
        return pd.DataFrame(columns=["Data", "Value", "Relevance"])
      return df
  '''
  Decided against filtering tables with AI, not worth the risk of missing data for what it gives
  Also takes a lot longer and requires more calls to OpenAI ($$$)
  def filter_tables(self, questions_list, pmcid, last_slash):
    xlname = pmcid + ".xlsx"
    xlpath = os.path.join(self.runpath[:last_slash-9], "tables", xlname)

    if not os.path.isfile(xlpath):
      return

    tables = pd.read_excel(xlpath, sheet_name=None)

    tablelist = []

    questions = "\n".join(questions_list)
    instructions = """
      \nThis is a table from the paper. Pull out any information that has to do with the four primary questions.
      Do not include any citations or references.Add a new column next to the extracted data called "Relevance"
      and explain how the data is relevant to the PK model. Return a pipe-delimeted table where every cell is
      wrapped in double quotes. Do not include code fences or commentary.Do not return the first column. As a reminder,
      here are the four primary questions the information should relate to:
    """
    instructions += questions

    for name, frame in tables.items():
      tablepair = f"{name}:\n{frame.to_string(index=False)}"
      tablepair += instructions

      response = self.llm_client.chat.completions.create(
        model=self.deployment,
        messages=[
          {"role": "user", "content": tablepair}
        ]
      )

      tablelist.append(response.choices[0].message.content)

    # will replace the original excel table
    with pd.ExcelWriter(xlpath, engine="openpyxl", mode="w") as xl:
      wrote_any = False
      for i, table in enumerate(tablelist):
        df = self.clean_output(table, pmcid)
        if not df.empty:
          df.to_excel(xl, sheet_name=f"Table {i+1}", index=False)
          wrote_any = True

      if not wrote_any:
        pd.DataFrame({"Message": [f"No parsable tables for {pmcid}"]}).to_excel(xl, sheet_name="EMPTY", index=False)
  '''

  # helper function to split too long fulltexts up
  def split_chunks(self, text_list, instructions, max_chunk_chars=30000):
    chunks = []
    cur = []
    length = 0
    for text in text_list:
      if length + len(text) > max_chunk_chars:
        cur.append(instructions)
        chunks.append("\n".join(cur))
        cur = []
        length = 0
      cur.append(text)
      length += len(text)

    if cur:
      cur.append(instructions)
      chunks.append("\n".join(cur))
    return chunks


  # function to handle freetext data processing
  def process_freetext(self, questions_list, pmcid, last_slash):

    # build the prompt
    # loop through and ask each question individually, then concat the outputs
    dfs = []
    for i in range(len(questions_list)):

      instructions = """
        Extract ONLY information from the paper freetext that is relevant to one of the four primary questions.
        Return your answer in a pipe-delimited (|) CSV file. Do not include any extra commentary or code
        fences. Use the following columns only: "Data" | "Value" | "Relevance". Make each piece of data its own row. 
        For the "relevance" column do not just restate the question, explain how it is relevant in the context of the paper.
        Get every piece of information regarding each question, be very detailed.

        Here is the question:
      """

      instructions += questions_list[i]

      # feed prompt to model
      text = instructions

      # remove disallowed special tokens
      text = re.sub(r"<\|.*?\|>", "", text)

      # ensure under maximum amount of tokens
      enc = tiktoken.encoding_for_model(os.getenv("AZURE_OPENAI_DEPLOYMENT"))
      ntokens = len(enc.encode(text))
      if ntokens > 128000:
        # too many tokens, split into chunks
        chunks = self.split_chunks(fulltext, instructions)
      else:
        chunks = [text]

      # for each chunk process and add to dataframe list, then concat them
      dataframes = []
      err = None
      for chunk in chunks:
        try:
          response = self.llm_client.chat.completions.create(
            model=self.deployment,
            messages=[
              {"role": "user", "content": chunk}
            ]
          )
          out = response.choices[0].message.content
        except Exception as e:
          print(f"[ERROR] Prompt failed")
          print(f"[ERROR] Exception: {e}")
          out = None
          err = str(e)

        if out is None:
          df = pd.DataFrame([["OpenAI error", err or "Unknown Error", questions_list[i]]], columns=["Data", "Value", "Relevance"])
        else:
          df = self.clean_output(out, pmcid)

        dataframes.append(df)

      df = pd.concat(dataframes, axis=0)

      try:
        df.columns = ["Data", "Value", "Relevance"] # make sure names are good
      except ValueError as e:
        print(f"[WARN] error renaming freetext columns for {pmcid}: {e}")
      dfs.append(df)

    # join together the outputs from each question
    final_output = pd.concat(dfs, axis=0)

    # get path of .xlsx file to write to
    xlname = pmcid + ".xlsx"
    xlpath = os.path.join(self.runpath[:last_slash-9], "tables", xlname)

    # write the returned table to a new sheet in the corresponding xlsx file
    if os.path.exists(xlpath):
      with pd.ExcelWriter(xlpath, engine="openpyxl", mode="a") as xl:
        final_output.to_excel(xl, sheet_name="Freetext Data", index=False)
    else:
      with pd.ExcelWriter(xlpath, engine="openpyxl", mode="w") as xl:
        final_output.to_excel(xl, sheet_name="Freetext Data", index=False)
    print("  >data extracted from freetext")

  # call to run the entire process
  def run(self):

    questions_list = [
      "How was the PK study designed, such as patient sample size, blood/tissue sampling time points, and patient population?",
      "What was the model, such as 1-compartment, 2-compartment, physiologically based model, or IVIVE?",
      "What are the parameters and their values, and where were they estimated from?",
      "Are there any reported factors in influencing pharmacokinetics?"
    ]
    last_slash = self.runpath.rindex("/")
    last_dot = self.runpath.rindex(".")
    pmcid = self.runpath[last_slash+1:last_dot]

    print("Paper:", pmcid)

    # 1) prime llm, get title and freetext from article to give context
    tags = ["article-meta/title-group", "p"]
    code = self.prime_llm(tags)
    if code == -1:
      return

    # 2) give the llm the tables and extract only information about the models
#    self.filter_tables(questions_list, pmcid, last_slash)
#    print("  >tables filtered")

    # 3) pull information from freetext
    self.process_freetext(questions_list, pmcid, last_slash)




