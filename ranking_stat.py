import pandas as pd
import numpy as np
import os
import argparse
import sys
import glob
import getopt
list_measures = ['mean dice','Mean DSC SegUnc', 'F1 score','F1 seg score','absolute difference','absolute volume difference']

def add_suffix_col(df, suffix):
    list_columnsr = df.columns
    list_newcolr = []
    for f in list_columnsr:
        if 'team' not in f:
            f+=suffix
        list_newcolr.append(f)
    df.columns = list_newcolr
    return df

def main(argv):

    parser = argparse.ArgumentParser(description='Aggregation  procedure')
    parser.add_argument('-rank_file', dest='rank', metavar='file with the ranking of results',
                        type=str, required=True,
                        help='File to read the ranking from')
    parser.add_argument('-aggreg_file', dest='aggregated', metavar='file with the database of results',
                        type=str, required=True,
                        help='File to read the ranking from')
    parser.add_argument('-path_out', dest='path_out', default=0.05,
                        type=str)
    parser.add_argument('-name_out', dest='name_out', type=str, help='name of the output file', default='ToWeb')

    

    try:
        args = parser.parse_args(argv)

    except getopt.GetoptError:
        print('ranking_stat.py -rank_file <ranking_file>  -aggreg_file <aggregation file> -name_out <name> -path_out <path where to save completed output> ')
        sys.exit(2)

    df_rank = pd.read_csv(args.rank)
    df_stat = pd.read_csv(args.aggregated)
    list_meas = [c for c in df_stat.columns if c in list_measures]
    df_statn = df_stat.groupby('team')[list_meas].quantile([0.5,0.25,0.75]).reset_index()
    df_quant50 = df_statn[df_statn['level_1']==0.50]
    df_quant25 = df_statn[df_statn['level_1']==0.25]
    df_quant75 = df_statn[df_statn['level_1']==0.75]
    df_rank = add_suffix_col(df_rank, '_rank')
    df_quant50 = add_suffix_col(df_quant50, '_50')
    df_quant25 = add_suffix_col(df_quant25, '_25')
    df_quant75 = add_suffix_col(df_quant75, '_75')
    
    df_merged  = pd.merge(df_quant50, df_quant25, on='team',how='left')
    
    df_merged2 = pd.merge(df_merged, df_quant75, on='team',how='left')
    
    df_mergedfin = pd.merge(df_merged2, df_rank, on='team',how='left')
    columns_to_keep = [c for c in df_mergedfin.columns if 'level' not in c and 'bootstrap' not in c]
    df_fin = df_mergedfin[columns_to_keep]
    name_new = args.path_out + os.path.sep + args.name_out
    df_fin.to_csv(name_new)
    
    
if __name__ == "__main__":
    # rank_file = '/Users/csudre/Documents/Documents_McBkp/Challenge/EvaluationResults/RankingPVS_Fin.csv'
    # aggreg_file = '/Users/csudre/Documents/Documents_McBkp/Challenge/EvaluationResults/AggregatedPVS.csv'
    # name_out = 'FinalToWebPVS.csv'
    # path_out = '/Users/csudre/Documents/Documents_McBkp/Challenge/'
    # list_argv = ['-rank_file',rank_file,'-aggreg_file',aggreg_file,'-name_out',name_out,'-path_out',path_out]
    # print(sys.argv[1:])
    # main(list_argv)
    
    main(sys.argv[1:])