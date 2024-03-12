import numpy as np
import os
import argparse

def file_loader():
    parser = argparse.ArgumentParser(description = "Loading and printing array") # initialize parser object
    parser.add_argument("--input", # add argument to the parser
                        "-i", # short format
                        required = True,
                        help="Filepath to CSV for loading and printing")
    args = parser.parse_args()
    return args

def process(filename):
    data = np.loadtxt(filename, delimiter=",")
    print(data)


def main():
    args = file_loader()
    filename = os.path.join("..",
                            "..",
                            "..",
                            "cds-visual",
                            "data",
                            "sample-data",
                            args.input)
    process(filename)

if __name__=="__main__": #if it's executed from the command line run the function "main", otherwise do NOTHING
    main()