import os
import pandas as pd
import pyodbc


def load_access_tables(dataset_path, tables):
    if len([x for x in pyodbc.drivers() if x.startswith('Microsoft Access Driver')]) == 0:
        raise Exception(
            'You need to install "Access Data Engine" depending your office X32 or X64 it might become challenging to install it.')
    path, file_name = os.path.split(dataset_path)
    df = {}

    connecion_string = 'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={0};'.format(dataset_path)

    cnxn = pyodbc.connect(connecion_string)

    for table_name in tables:
        path_pickle = "{}/{}.pkl".format(path, table_name)

        if os.path.exists(path_pickle):
            print("Loaded from pickle: {}".format(table_name))
            df[table_name] = pd.read_pickle(path_pickle)
            pass
        else:
            query = "SELECT * FROM {}".format(table_name)

            df[table_name] = pd.read_sql(query, cnxn)

            df[table_name].to_pickle(path_pickle)
            pass

        print("Loaded {} --> Table: {} | Number of rows: {}".format(
            dataset_path, table_name, len(df[table_name])))

        pass

    cnxn.close()

    return df
    pass


if __name__ == "__main__":
    load_access_tables(
        "D:/Dropbox (CCNR)/Ravandi, Babak/datasets/FoodData_Central/2020-03-31/FoodData_Central_2020-03-31.accdb",
        ["branded_food"])
    pass
