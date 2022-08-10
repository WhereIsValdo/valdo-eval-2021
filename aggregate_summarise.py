import pandas as pd
import numpy as np
import os
import argparse
import sys
import glob
import getopt

list_measures=['F1 score', 'mean dice','absolute difference','absolute volume difference',
                'F1 seg score','Mean DSC SegUnc']

def aggregate_files(list_files,cut='PairwiseMeasure_'):
    list_df = []
    for f in list_files:
        #print(cut)
        name = f.split(cut)[1]
        team = name.split('_')[0]
        df_tmp = pd.read_csv(f)
        df_tmp['team'] = team
        list_df.append(df_tmp)
    df_aggregated = pd.concat(list_df)
    return df_aggregated

def summary_stats(df_aggregated):
    list_existing_measures = [c for c in df_aggregated.columns if c in list_measures]
    table_fin = df_aggregated.groupby('team')[list_existing_measures].quantile([0.5,0.25,0.75])
    return table_fin




def main(argv):

    parser = argparse.ArgumentParser(description='Completion  procedure')
    parser.add_argument('-reg_exp', dest='reg_exp', metavar='file with the database of results',
                        type=str, required=True,
                        help='File to read the data from')
    parser.add_argument('-path_out', dest='path_out', default=0.05,
                        type=str)
    parser.add_argument('-name_out', dest='name_out', type=str, help='name of the output file', default='Completed')

    

    try:
        args = parser.parse_args(argv)

    except getopt.GetoptError:
        print('ranking_comparison.py -id <ids with ref> -f <file to complete> -path_out <path where to save completed output> -prefix <prefix to add to indicate completion> ')
        sys.exit(2)

    list_files = glob.glob(args.reg_exp)
    print(len(list_files), args.reg_exp)
    df_agg = aggregate_files(list_files)
    df_sum = summary_stats(df_agg)
    df_agg.to_csv(args.path_out + '/' + args.name_out + '.csv')
    df_sum.to_csv(args.path_out + '/' + args.name_out +'_stat.csv')
    
if __name__ == "__main__":
    main(sys.argv[1:])