from Bio import Entrez as ez
from paper import Abstract
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from math import ceil
import torch, time, io, os, gc, sys
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead
from datetime import datetime

DEFAULT_SEARCH = "pharmacokinetics models"
DEFAULT_RETMAX = 100000 # largest allowed by pubmed
DEFAULT_THRESHOLD = 0.65 # .55 to .65 is good range depending on if you want more false positives or false negatives
DEFAULT_BATCH_SIZE = 20
DEFAULT_FROM_DATE = "1781/01/01" # date of earliest pubmed publication
DEFAULT_TO_DATE = datetime.today().strftime("%Y/%m/%d") # today's date

class ScreenAbstracts:

  '''
    Class that defines pubmed searching and abstract
    screening from user input to find PK articles
  '''

  def __init__(self, search, retmax, batch_size, gpumax_bytes, from_date, to_date, threshold=DEFAULT_THRESHOLD, save_to=None, model_name="./best_f1_model"):

    # set up LLM
    self.model_name = model_name

    if torch.cuda.is_available():
      total_mem = torch.cuda.get_device_properties(0).total_memory
      if total_mem < 2 * 1024 ** 3: # if total gpu mem < 6 gb, do cpu
        self.device = torch.device("cpu")
      else:
        self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
    self.model.eval()

    self.gpumax_bytes = gpumax_bytes

    # set values
    if search == "":
      self.search = DEFAULT_SEARCH
    else:
      self.search = search

    if retmax == "":
      self.retmax = DEFAULT_RETMAX
    else:
      self.retmax = int(retmax)

    self.threshold = DEFAULT_THRESHOLD

    if batch_size == "":
      self.batch_size = DEFAULT_BATCH_SIZE
    else:
      self.batch_size = int(batch_size)

    if save_to == None:
      print("Err: need to assign directory to save to")
      return
    else:
      self.save_to = save_to

    if from_date == "":
      self.from_date = DEFAULT_FROM_DATE
    else:
      formats = ["%Y/%m/%d", "%Y/%m", "%Y"]
      VALID = 0
      for fmt in formats:
        try:
          datetime.strptime(from_date, fmt)
          VALID = 1
          break
        except ValueError as e:
          continue

      if VALID == 0:
        print(f"'From' date is in improper format: {from_date} does not match any '%Y/%m/%d', '%Y/%m', or '%Y' formats")
        sys.exit(1)

      self.from_date = from_date

    if to_date == "":
      self.to_date = DEFAULT_TO_DATE
    else:
      formats = ["%Y/%m/%d", "%Y/%m", "%Y"]
      VALID = 0
      for fmt in formats:
        try:
          datetime.strptime(to_date, fmt)
          VALID = 1
          break
        except ValueError as e:
          continue

      if VALID == 0:
        print(f"'To' date is in improper format: {to_date} does not match any '%Y/%m/%d', '%Y/%m', or '%Y' formats")
        sys.exit(1)

      self.to_date = to_date

  # function to reload the model, mostly to reset memory usage
  def reload_model(self):
    del self.model
    gc.collect()
    torch.cuda.empty_cache()
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
    self.model.eval()

  # wrap fetch.read() to handle errors and retry on fail
  def safe_read(self, fetch, max_retries=3):
    for attempt in range(max_retries):
      try:
        return fetch.read()
      except IncompleteRead as e:
        print(f"[WARN] IncompleteRead on attempt {attempt+1}/{max_retries}, retrying...")
        time.sleep(1 + attempt)
    print("Entrez read failure")


  # helper function to chunk up longer abstracts
  def pred_chunks(self, text, tokenizer):

    max_len = self.model.config.max_position_embeddings
    stride = max_len // 2

    tokenized = tokenizer(
      text,
      truncation=True,
      padding="max_length",
      max_length=max_len,
      return_overflowing_tokens=True,
      return_attention_mask=True,
      stride=stride,
      return_tensors="pt"
    )

    with torch.no_grad():
      logits = self.model(
        input_ids=tokenized.input_ids.to(self.device),
        attention_mask=tokenized.attention_mask.to(self.device)
      ).logits

    return logits.mean(dim=0, keepdim=True)

  # function to run prediction of each abstract
  def predict(self, abstract_list, valid_pmids, threshold, tokenizer):
    # for each item in list, predict if PK
    for i, abstr in enumerate(abstract_list):
      final_logits = self.pred_chunks(abstr.text, tokenizer).detach().cpu()
      probs = torch.softmax(final_logits, dim=1)
      pred_class = torch.argmax(final_logits, dim=1)

      # 1 == is PK, 0 == not PK
      prob_PK = probs[0, 1].item()
      abstr.prob = float(prob_PK)
      if abstr.prob > threshold:
        abstr.isPK = True
        valid_pmids.append(abstr.pmid)
      print (f"{abstr}")

      # after prediction delete abstract data and clean up
      torch.cuda.empty_cache()
      del final_logits, probs, abstr.text

  def link_to_pmc(self, valid_pmids):
    pmcids = []
    for pmid in valid_pmids:
      try:
        for attempt in range(3):
          linker = ez.elink(
            dbfrom="pubmed",
            db="pmc",
            id=pmid
          )
          link_res = ez.read(linker)
          pmcid_num = (link_res[0].get("LinkSetDb") or [{}])[0].get("Link", [{}])[0].get("Id")
          if pmcid_num:
            pmcid = f"PMC{pmcid_num}"
            print(f"Found PMC article (PMCID = {pmcid}) for abstract PMID = {pmid}")
            pmcids.append(pmcid)

          time.sleep(.1) # slow down request rate to limit to
          break; # break from inner loop
      except Exception as e:
        print(f"[WARN] error fetching PMC for PMID: {pmid}: {str(e)}")
        print(f"[WARN] retrying, error on {attempt+1}/3 attempts")
        time.sleep(.35 + attempt)

    return pmcids

  def search_pubmed(self, query, nresults):
    search = ez.esearch(
      db="pubmed",
      term=query,
      retmax=nresults,
      retmode="xml",
      mindate=self.from_date,
      maxdate=self.to_date,
      datetype="pdat" # publication date
    )
    return ez.read(search)["IdList"]

  def fetch_abstracts(self, pmids):
    for attempt in range(3):
      try:
        fetch = ez.efetch(
          db="pubmed",
          id=",".join(pmids),
          rettype="abstract",
          retmode="text"
        )
        abstracts = self.safe_read(fetch)
        fetch.close()

        abstract_list = abstracts.split("\n\n\n")
        for i, text in enumerate(abstract_list):
          abstract = Abstract(text)
          abstract.pmid = pmids[i]
          abstract_list[i] = abstract

        return abstract_list
      except Exception as e:
        print(f"[WARN] Error fetching abstracts on attempt {attempt}/3, retrying...")
        time.sleep(.1 + attempt)
    print("Entrez fetch failure")

  # helper function to save each xml fulltext
  def save_xml(self, pmcid, xml_text, dirpath):

    os.makedirs(dirpath, exist_ok=True)

    filename = pmcid + ".xml"
    filepath = os.path.join(dirpath, filename)
    with open(filepath, "wb") as file:
      file.write(xml_text)

  def fetch_and_save_xmls(self, pmcids):
    dirpath = self.save_to + "/fulltexts"
    os.makedirs(dirpath, exist_ok=True)
    for pmcid in pmcids:
      try:
        pmcid_num = pmcid.replace("PMC", "")
        fetch = ez.efetch(
          db="pmc",
          id=pmcid_num,
          rettype="xml",
          retmode="xml"
        )
        xml_text = self.safe_read(fetch)
        fetch.close()

        print(f"saving {pmcid} to {dirpath}")
        self.save_xml(pmcid, xml_text, dirpath)
        del xml_text, pmcid
        gc.collect()
        time.sleep(.2) # slow down requests
      except HTTPError as e:
        print(f"error fetching {pmcid}: {e.code} {e.reason}")
        time.sleep(.35) # slow down requests
        continue
      except URLError as e:
        print(f"error fetching {pmcid}: {e.reason}")
        time.sleep(.35) # slow down requests
        continue
      except Exception as e:
        print(f"error fetching {pmcid}: {str(e)}")
        time.sleep(.35)
        continue
    return len(os.listdir(dirpath))

  def run(self):
    ez.email = os.getenv("ENTREZ_EMAIL")
    ez.api_key = os.getenv("ENTREZ_API_KEY")

    print("Running on device:", self.device)

    # 1) search pubmed and get pmids
    pmids = self.search_pubmed(self.search, self.retmax)

    # limit fetch and predict to batches of BATCH_SIZE
    # to prevent memory running out
    valid_pmids = []
    for start in range(0, len(pmids), self.batch_size):

      # monitor gpu memory usage, if too high reload the model
      if torch.cuda.memory_allocated() > self.gpumax_bytes:
        self.reload_model()

      chunk = pmids[start:start+self.batch_size]
      abstract_chunk = self.fetch_abstracts(chunk)
      self.predict(abstract_chunk, valid_pmids, self.threshold, self.tokenizer)
      del abstract_chunk[:]
      del abstract_chunk
      self.reload_model()


    # get pmcids
    print()
    pmcids = self.link_to_pmc(valid_pmids)
    print()

    # COMMENT BLOCK IF DOING FULL SCREENING
    # this block just saves the pmcids to a file so just comment out the return if
    # you still want those when doing a full screening
    '''
    npapers = len(valid_pmids)
    path = self.save_to + "/pk_pmids.txt"
    with open(path, "w") as file:
      file.write("Pubmed PK Papers:\n")
    for pmcid in pmcids:
      with open(path, "a") as file:
        file.write(pmcid + "\n")
    return npapers
    '''

    # search pmc for papers, save as xml file
    return self.fetch_and_save_xmls(pmcids)
