from env import host, user, password

url = f'mysql+pymysql://{user}:{password}@{host}/zillow'
###Wrangle
def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_data_from_mysql():
    query = '''
    SELECT *
    FROM customers
    JOIN internet_service_types USING (internet_service_type_id)
    WHERE contract_type_id = 3

def clean_data(df):
    df = df[['customer_id', 'total_charges', 'monthly_charges', 'tenure']]
    df.total_charges = df.total_charges.str.strip().replace('', np.nan).astype(float)
    df = df.dropna()
    return df


def plot_residuals(x, y):
    '''
    Plots the residuals of a model that uses x to predict y. Note that we don't
    need to make any predictions ourselves here, seaborn will create the model
    and predictions for us under the hood with the `residplot` function.
    '''
    return sns.residplot(x, y)


#split Scale
def split_my_data(data, train_ratio=.80, seed=123):
    '''the function will take a dataframe and returns train and test dataframe split 
    where 80% is in train, and 20% in test. '''
    return train_test_split(data, train_size = train_ratio, random_state = seed)


def standard_scaler(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled


def uniform_scaler(train, test, seed=123):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=seed, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

##Visual
def evaluation_example1(df, x, y):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='dimgray')

Evaluate