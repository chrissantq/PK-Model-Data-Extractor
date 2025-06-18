import sys, io

class OutTee:
  def __init__(self, *streams):
    self.streams = streams
  def write(self, data):
    for s in self.streams:
      s.write(data)
  def flush(self):
    for s in self.streams:
      s.flush()

class InTee(io.TextIOBase):
  def __init__(self, stdin, logfile):
    self.stdin = stdin
    self.log = logfile

  def readline(self, *args):
    line = self.stdin.readline(*args)
    self.log.write(line)
    self.log.flush()
    return line

  def read(self, *args):
    data = self.stdin.read(*args)
    self.log.write(data)
    self.log.flush()
    return data

  def __getattr__(self, attr):
    return getattr(self.stdin, attr)
