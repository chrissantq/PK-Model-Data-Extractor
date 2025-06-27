from io import StringIO
import tabfetch as tf
import pandas as pd
import os


class GetTables:

  '''
    Class to define object to screen xml files for
    tables and output those tables into .xlsx files
  '''

  def __init__(self, rootpath):
    self.rootpath = rootpath

  # function to save dataframe to a excel file
  def save_tables(self, df_list, excelpath):

    # map sheet names to tables (dataframes)
    df_map = {}
    for i, df in enumerate(df_list):
      sheet_name = "Table " + str(i+1)
      df_map[sheet_name] = df

    # write to excel file
    with pd.ExcelWriter(excelpath, engine="openpyxl") as writer:
      for name, df in df_map.items():
        # flatten so it doesnt have to index for some reason idk
        if isinstance(df.columns, pd.MultiIndex):
          df.columns = ['_'.join(map(str, col)).strip() for col in df.columns]
        df.to_excel(writer, sheet_name=name, index=False)

  def run(self):

    storepath = os.path.join(self.rootpath, "tables")
    os.makedirs(storepath, exist_ok=True)

    seen = set()
    read_path = os.path.join(self.rootpath, "fulltexts")
    xml = tf.read_file(read_path)
    while xml and xml not in seen:
      pmcid = xml.pmcid

      excelname = pmcid + ".xlsx"
      excelpath = os.path.join(storepath, excelname)

      if os.path.exists(excelpath):
        print(f"Paper {pmcid} tables have already been written")
        seen.add(xml)
        xml = xml.next
        continue

      t = xml.tab_head
      df_list = []
      while t:
        text = t.text
        try:
          df = pd.read_html(StringIO(text))[0]
        except ValueError:
          print(f"No tables found for {pmcid}, skipping")
          t = t.next
          continue


        df_list.append(df)
        t = t.next

      # save dataframes as individual sheets in excel file

      if not df_list:
        print(f"No tables found for {pmcid}, skipping")
      else:
        self.save_tables(df_list, excelpath)
        print(f"Paper {pmcid} tables written to {excelpath}")

      seen.add(xml)
      xml = xml.next




