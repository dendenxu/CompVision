import coloredlogs
import argparse
import matplotlib as mpl


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


# initailization for plotting and logging
# Setting up font for matplotlib
mpl.rc("font", family=["Josefin Sans", "Trebuchet MS", "Inconsolata"], weight="medium")

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.
