#!/usr/bin/env python


import yaml
import argparse 

from bolo import Top

def main():
    """Hook for setup.py"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", default=None, required=True,
                        help="Input configuration file")
    parser.add_argument('-o', "--output", default=None, 
                        help="Output file")

    args = parser.parse_args()
    
    dd = yaml.safe_load(open(args.input))
    top = Top(**dd)
    top.run()
    top.instrument.print_summary()
    top.instrument.print_optical_output()

    if args.output:
        top.instrument.write_tables(args.output)

if __name__ == '__main__':
    main()
