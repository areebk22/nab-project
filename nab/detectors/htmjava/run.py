#!/usr/bin/env python
# Copyright 2014-2015 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import argparse
import os
try:
  import simplejson as json
except ImportError:
  import json

from nab.runner import Runner
from nab.util import checkInputs
from nab.detectors.htmjava.htmjava_detector import HtmjavaDetector


def get_nth_parent_dir(n, path):
  """
  Return the Nth parent of `path` where the 0th parent is the direct parent
  directory.
  """
  parent = os.path.dirname(path)
  if n == 0:
    return parent

  return get_nth_parent_dir(n-1, parent)

def main(args):

  filepath = os.path.realpath(__file__)

  # Find the main NAB folder
  # Assuming `filepath` is ~ <...>/NAB/nab/detectors/htmjava/run.py
  root = get_nth_parent_dir(3, filepath)

  numCPUs = int(args.numCPUs) if args.numCPUs is not None else None

  dataDir = os.path.join(root, args.dataDir)
  windowsFile = os.path.join(root, args.windowsFile)
  resultsDir = os.path.join(root, args.resultsDir)
  profilesFile = os.path.join(root, args.profilesFile)

  runner = Runner(dataDir=dataDir,
                  labelPath=windowsFile,
                  resultsDir=resultsDir,
                  profilesPath=profilesFile,
                  numCPUs=numCPUs)

  runner.initialize()

  runner.detect({'htmjava': HtmjavaDetector})


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("--skipConfirmation",
                    help="If specified will skip the user confirmation step",
                    default=False,
                    action="store_true")

  parser.add_argument("--dataDir",
                    default="data",
                    help="This holds all the label windows for the corpus.")

  parser.add_argument("--resultsDir",
                    default="results",
                    help="This will hold the results after running detectors "
                    "on the data")

  parser.add_argument("--windowsFile",
                    default=os.path.join("labels", "combined_windows.json"),
                    help="JSON file containing ground truth labels for the "
                         "corpus.")

  parser.add_argument("-p", "--profilesFile",
                    default=os.path.join("config", "profiles.json"),
                    help="The configuration file to use while running the "
                    "benchmark.")

  parser.add_argument("-n", "--numCPUs",
                    default=None,
                    help="The number of CPUs to use to run the "
                    "benchmark. If not specified all CPUs will be used.")

  args = parser.parse_args()

  if args.skipConfirmation or checkInputs(args):
    main(args)
