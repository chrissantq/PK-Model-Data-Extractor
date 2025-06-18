from Bio import Entrez as ez
from paper import Abstract
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from math import ceil
import torch, time, io, os
from urllib.error import HTTPError, URLError

DEFAULT_SEARCH = "PK Model"
DEFAULT_RETMAX = 25
DEFAULT_THRESHOLD = 0.96
DEFAULT_BATCH_SIZE = 25

class ScreenAbstracts:

  '''
    Class that defines pubmed searching and abstract
    screening from user input to find PK articles
  '''

  def __init__(self, search, retmax, threshold, batch_size, save_to=None):
    if search == "":
      self.search = DEFAULT_SEARCH
    else:
      self.search = search

    if retmax == "":
      self.retmax = DEFAULT_RETMAX
    else:
      self.retmax = int(retmax)

    if threshold == "":
      self.threshold = DEFAULT_THRESHOLD
    else:
      self.threshold = float(threshold)

    if batch_size == "":
      self.batch_size = DEFAULT_BATCH_SIZE
    else:
      self.batch_size = int(batch_size)

    if save_to == None:
      print("Err: need to assign directory to save to")
      return
    else:
      self.save_to = save_to

  # helper function to chunk up longer abstracts
  def pred_chunks(self, text, tokenizer, model):

    max_len = model.config.max_position_embeddings
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

    logits = model(
      input_ids=tokenized.input_ids,
      attention_mask=tokenized.attention_mask
    ).logits

    return logits.mean(dim=0, keepdim=True)

  # function to run prediction of each abstract
  def predict(self, abstract_list, threshold):
    model_name = "./pk_vs_not_final"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # for each item in list, predict if PK
    for i, abstr in enumerate(abstract_list):
      final_logits = self.pred_chunks(abstr.text, tokenizer, model)
      probs = torch.softmax(final_logits, dim=1)
      pred_class = torch.argmax(final_logits, dim=1)

      # 1 == is PK, 0 == not PK
      prob_PK = probs[0, 1].item()
      abstr.prob = prob_PK
      if prob_PK > threshold: # 96% seemed to be a good threshold
        abstr.isPK = True

      # after prediction don't need text (takes of lots of memory)
      abstr.text = ""

  def link_to_pmc(self, abstract_list):
    for abstr in abstract_list:
      linker = ez.elink(
        dbfrom="pubmed",
        db="pmc",
        id=abstr.pmid
      )
      link_res = ez.read(linker)
      pmcid_num = (link_res[0].get("LinkSetDb") or [{}])[0].get("Link", [{}])[0].get("Id")
      if pmcid_num:
        abstr.pmcid = f"PMC{pmcid_num}"

      time.sleep(.15) # slow down request rate to limit to ~10/sec

  def search_pubmed(self, query, nresults):
    search = ez.esearch(
      db="pubmed",
      term=query,
      retmax=nresults,
      retmode="xml"
    )
    return ez.read(search)["IdList"]

  def fetch_abstracts(self, pmids):
    fetch = ez.efetch(
      db="pubmed",
      id=",".join(pmids),
      rettype="abstract",
      retmode="text"
    )
    abstracts = fetch.read()
    fetch.close()

    abstract_list = abstracts.split("\n\n\n")
    for i, text in enumerate(abstract_list):
      abstract = Abstract(text)
      abstract.pmid = pmids[i]
      abstract_list[i] = abstract

    return abstract_list

  # helper function to save each xml fulltext
  def save_xml(self, abstr, xml_text, dirpath):

    os.makedirs(dirpath, exist_ok=True)

    filename = abstr.pmcid + ".xml"
    filepath = os.path.join(dirpath, filename)
    with open(filepath, "wb") as file:
      file.write(xml_text)

  def fetch_and_save_xmls(self, abstract_list):
    dirpath = self.save_to + "/fulltexts"
    os.makedirs(dirpath, exist_ok=True)
    for abstr in abstract_list:
      try:
        fetch = ez.efetch(
          db="pmc",
          id=abstr.pmcid,
          rettype="full",
          retmode="xml"
        )
        xml_text = fetch.read()
        fetch.close()

        print(f"saving {abstr.pmcid} to {dirpath}")
        self.save_xml(abstr, xml_text, dirpath)
        time.sleep(.15) # slow down requests
      except HTTPError as e:
        print(f"error fetching {abstr.pmcid}: {e.code} {e.reason}")
        time.sleep(.15) # slow down requests
        continue
      except URLError as e:
        print(f"error fetching {abstr.pmcid}: {e.reason}")
        time.sleep(.15) # slow down requests
        continue
    return len(os.listdir(dirpath))

  def run(self):
    ez.email = os.getenv("ENTREZ_EMAIL")
    ez.api_key = os.getenv("ENTREZ_API_KEY")

    # 1) search pubmed and get pmids
    pmids = self.search_pubmed(self.search, self.retmax)

    # limit fetch and predict to batches of BATCH_SIZE
    # to prevent memory running out
    abstract_list = []
    for start in range(0, len(pmids), self.batch_size):
      chunk = pmids[start:start+self.batch_size]
      abstract_chunk = self.fetch_abstracts(chunk)
      self.predict(abstract_chunk, self.threshold)
      # filter out non pk papers and add remaining to main list
      abstract_chunk = [ab for ab in abstract_chunk if ab.isPK == True]
      abstract_list.extend(abstract_chunk)

    # get pmcids
    self.link_to_pmc(abstract_list)

    # filter out articles with no pmcid
    abstract_list = [ab for ab in abstract_list if ab.pmcid != None]

    # search pmc for papers, save as xml file
    return self.fetch_and_save_xmls(abstract_list)


