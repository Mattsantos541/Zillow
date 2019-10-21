from env import host, user, password
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
url = f'mysql+pymysql://{user}:{password}@{host}/zillow'
###Wrangle
def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_data_from_mysql():
    df = pd.read_sql(
'''SELECT 
p17.transactiondate,p.id,p.bathroomcnt as bathrooms,p.bedroomcnt as bedrooms, `lotsizesquarefeet`,p.calculatedfinishedsquarefeet as sqft, p.taxvaluedollarcnt as tax_value, `regionidcity` as region_id
FROM propertylandusetype pl
JOIN
properties_2017 p ON p.propertylandusetypeid = pl.propertylandusetypeid
JOIN
predictions_2017 p17 ON p17.id = p.id
WHERE 
p.propertylandusetypeid in (279,261) 
AND 
(p17.transactiondate LIKE '%%2017-05%%' or p17.transactiondate LIKE '%%2017-06%%')
AND
p.calculatedfinishedsquarefeet IS NOT NULL
and
p.bedroomcnt > 0
and 
p.bathroomcnt > 0
and
p.taxvaluedollarcnt > 0
and 
`lotsizesquarefeet` >0
and 
regionidcity > 0;'''
,url)
    return df


def taxcounty():
    df = pd.read_sql("""
    SELECT p.taxvaluedollarcnt as tax_value, p.fips, p.taxamount, round(p.taxamount/p.taxvaluedollarcnt,4) as tax_rate 
FROM propertylandusetype pl
JOIN
properties_2017 p ON p.propertylandusetypeid = pl.propertylandusetypeid
JOIN
predictions_2017 p17 ON p17.id = p.id
WHERE 
p.propertylandusetypeid in (279,261) 
AND 
(p17.transactiondate LIKE '%%2017-05%%' or p17.transactiondate LIKE '%%2017-06%%')
AND
p.calculatedfinishedsquarefeet IS NOT NULL
and
p.bedroomcnt > 0
and 
p.bathroomcnt > 0
and
p.taxvaluedollarcnt > 0
and
p.taxamount > 0;

""",url)
    return df