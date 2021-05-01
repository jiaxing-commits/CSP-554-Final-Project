''' Utility functions for spark_pkg/*/sql.py '''

def summarize_df(session,df,numeric_cols):
    """
    Summarize the dataframe attributes
    Args:
        session (SparkSession): active SparkSession
        df (DataFrame): dataframe of interest
        numeric_cols (list): list of numeric_attribute names
    """
    df.createOrReplaceTempView('dfsql')
    for col_str in df.schema.names:
        print('col:', col_str) 
        if col_str in numeric_cols:
            df.describe(col_str).show()
        else:
            dftmp = session.sql('select %s, count(*) as count from dfsql group by %s'%(col_str,col_str)).show()
    df.printSchema()
    df.show(20)

