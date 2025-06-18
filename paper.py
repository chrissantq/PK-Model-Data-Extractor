class Abstract:
  def __init__(self, text):
    self.text = text
    self.isPK = False
    self.pmid = None
    self.pmcid = None
    self.prob = 0

  def __str__(self):
    return f"<pmid: {self.pmid}, isPK: {self.isPK}, prob: {self.prob}>"
