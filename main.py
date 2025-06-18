from openai import AzureOpenAI
from dotenv import load_dotenv, dotenv_values
import os, datetime, time

from screen_abstracts import ScreenAbstracts
from get_tables import GetTables
from freetext_parse import FetchModelInformation

def main():

  load_dotenv()

  client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
  )
  DEPLOY = os.environ["AZURE_OPENAI_DEPLOYMENT"]

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
  dt = datetime.datetime.now()
  run_id = dt.strftime("%x") + "+" + dt.strftime("%X")
  run_id = "run_" + run_id.replace("/", "-")

  print("Run will be saved under:", run_id)
  print()

  # 1) Abstract screening and xml fetching

  pubmed_search = input("PubMed search: ")
  retmax = input("Number of PubMed results: ")
  thresh = input("How sure must the LLM be that abstract is PK (in decimal percentage, i.e. 0.98 -> 98% sure): ")
  batch_size = input("Size of processing batches: ")
  print()

  print("Screening abstracts and fetching fulltexts...")
  save_dir = "./run_outputs/" + run_id
  screener = ScreenAbstracts(
    search=pubmed_search,
    retmax=retmax,
    threshold=thresh,
    batch_size=batch_size,
    save_to=save_dir
  )
  num_papers = screener.run()
  print(f"Retrieved {num_papers} papers from PMC")

  # 2) Get tables from xml files

  print()
  print("Extracting tables...")
  extractor = GetTables(save_dir)
  extractor.run()


  # 3) Get model data from each paper

  print()
  print("Extracting model information from each paper...")
  fulltextdir = os.path.join(save_dir, "fulltexts")
  for i, paper in enumerate(os.listdir(fulltextdir)):
    print(f"Progress: {i}/{num_papers}")
    xmlpath = os.path.join(fulltextdir, paper)
    modeler = FetchModelInformation(xmlpath, client, DEPLOY)
    modeler.run()
    # break
  print(f"Progress: {num_papers}/{num_papers}")
  print("Done!")


if __name__ == "__main__":
  start = time.perf_counter()
  main()
  elapsed = time.perf_counter() - start

  h, r = divmod(elapsed, 3600)
  m, s = divmod(r, 60)

  print(f"Finished in {int(h):02d} hrs {int(m):02d} mins {s:06.3f} secs")



