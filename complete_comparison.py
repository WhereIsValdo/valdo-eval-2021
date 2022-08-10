import pandas as pd
import numpy as np
import argparse
import os
import sys
import getopt

dict_args = {'F1 score':0,
             'mean dice':0,
             'absolute difference':100000,
             'absolute volume difference':100000,
             'F1 seg score':0,
             'Mean DSC SegUnc':0}

def create_id(x):
    return x.split('_')[0]

def get_subscript_name(df_eval):
    example_name = df_eval['Name (ref)']


def identify_missing_cases(df_eval, df_id):
    df_eval['id'] = df_eval['Name (ref)'].apply(lambda x: create_id(x))
    list_exp = list(df_id['Name (ref)'])
    list_eff = list(df_eval['Name (ref)'])
    list_missing = [f for f in list_exp if f not in list_eff]
    return list_missing

def create_missing_rows(df_eval, list_missing):
    list_measures = [c for c in df_eval.columns if c in dict_args.keys()]
    list_rows = []
    for f in list_missing:
        new_dict = {}
        new_dict['Name (ref)'] = f
        new_dict['Name (seg)'] = ''
        for k in list_measures:
            new_dict[k] = dict_args[k]
        list_rows.append(new_dict)
    return pd.DataFrame.from_dict(list_rows)

def complete_team(df_eval, df_id):
    list_missing = identify_missing_cases(df_eval, df_id)
    if len(list_missing) == 0:
        return df_eval
    else:
        df_toadd = create_missing_rows(df_eval, list_missing)
        df_new = pd.concat([df_eval, df_toadd])
        df_new = df_new.reset_index(drop=True)
        df_new['id'] = df_new['Name (ref)'].apply(lambda x: create_id(x))
        return df_new

def main(argv):

    parser = argparse.ArgumentParser(description='Completion  procedure')
    parser.add_argument('-f', dest='file_in', metavar='file with the database of results',
                        type=str, required=True,
                        help='File to read the data from')
    parser.add_argument('-path_out', dest='path_out', default=0.05,
                        type=str)
    parser.add_argument('-id', dest='id_file', type=str, help='file with list of references')
    parser.add_argument('-prefix', dest='prefix', type=str, help='prefix to add to the file', default='Completed')

    

    try:
        args = parser.parse_args(argv)

    except getopt.GetoptError:
        print('ranking_comparison.py -id <ids with ref> -f <file to complete> -path_out <path where to save completed output> -prefix <prefix to add to indicate completion> ')
        sys.exit(2)

    df_eval = pd.read_csv(args.file_in)
    df_id = pd.read_csv(args.id_file)
    df_new = complete_team(df_eval, df_id)
    new_name = args.path_out + '/' + args.prefix + os.path.split(args.file_in)[1]

    df_new.to_csv(new_name)

if __name__ == "__main__":
    main(sys.argv[1:])
    


