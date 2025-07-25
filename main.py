from openai import AzureOpenAI
from dotenv import load_dotenv, dotenv_values
import os, sys, datetime, time, shutil

from screen_abstracts import ScreenAbstracts
from get_tables import GetTables
from freetext_parse import FetchModelInformation
from tee import InTee, OutTee

def main():

  '''
    1) Screen pubmed for abstracts based on user search

    2) Of found abstracts, fetch corresponding PMC article if exists

    3) For each returned PMC article, pull out information on the model
        - fetch xml
        - pull tables out from article
        - get title, freetext from article
        - contextualize the AI model for the paper
        - pull out data/info from freetext and tables
  '''

  '''
    0) Preliminary steps
      - load .env file
      - ready the AzureOpenAI client
      - build the run file tree
      - prepare the log file and configure to write to it
  '''

  load_dotenv()

  client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
  )
  DEPLOY = os.getenv("AZURE_OPENAI_DEPLOYMENT")

  pubmed_search = input("PubMed search: ")
  retmax = input("Number of PubMed results: ")
  batch_size = input("Size of processing batches: ")
  print()
  print("Date range -- enter in YYYY/MM/DD format, may also just enter YYYY, YYYY/MM, or just leave empty for defaults:")
  from_date = input("From: ")
  to_date = input("To: ")
  print()

  run_id = "run_" + from_date + "_to_" + to_date
  run_id = run_id.replace("/", "-")

  save_dir = os.path.join("./run_outputs", run_id)

  i = 1
  while True:
    temp_id = run_id
    if not os.path.exists(save_dir):
      break
    else:
      temp_id += "_" + str(i)
      save_dir = os.path.join("./run_outputs", temp_id)
      i += 1


  os.makedirs(save_dir, exist_ok=True)
  logfile = os.path.join(save_dir, "log.out")

  logfp = open(logfile, "w")
  sys.stdout = OutTee(sys.__stdout__, logfp)
  sys.stderr = OutTee(sys.__stderr__, logfp)
  sys.stdin = InTee(sys.__stdin__, logfp)

  print("Run will be saved under:", save_dir)
  print()

  # 1) Abstract screening and xml fetching

  print("Screening abstracts and fetching fulltexts...")
  screener = ScreenAbstracts(
    search=pubmed_search,
    retmax=retmax,
    batch_size=batch_size,
    gpumax_bytes=10 * 1024 ** 3, # 10 GB threshold for memory usage before reloading
    save_to=save_dir,
    from_date=from_date,
    to_date=to_date
  )
  num_papers = screener.run()
  print(f"Retrieved {num_papers} papers from PMC")
  
  # 2) Get tables from xml files

  print()
  print("Extracting tables...")
  # temp
  #save_dir = "./run_outputs/run_2018_to_2018/"
  extractor = GetTables(save_dir)
  extractor.run()

  #num_papers = "2411"

  # 3) Get model data from each paper

  print()
  print("Extracting model information from each paper...")
  fulltextdir = os.path.join(save_dir, "fulltexts")
  sorted_files = sorted(os.listdir(fulltextdir), key=lambda f: int(f[3:-4]))
  for i, paper in enumerate(sorted_files):
    print(f"Progress: {i}/{num_papers}")
    xmlpath = os.path.join(fulltextdir, paper)
    modeler = FetchModelInformation(xmlpath, client, DEPLOY)
    modeler.run()
    # break
  print(f"Progress: {num_papers}/{num_papers}")

  # 4) delete fulltexts
  if os.path.exists(fulltextdir):
    try:
      shutil.rmtree(fulltextdir)
    except OSError as e:
      print(f"Err deleting fulltexts: {e}")
  else:
    print("Err: fulltext directory doesn't exist")

  print("Done!")

if __name__ == "__main__":
  start = time.perf_counter()
  main()
  elapsed = time.perf_counter() - start

  h, r = divmod(elapsed, 3600)
  m, s = divmod(r, 60)

  print(f"Finished in {int(h):02d} hrs {int(m):02d} mins {s:06.3f} secs")



