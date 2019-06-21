
import numpy as np
import pandas as pd
import os

def read_data():
    url='C:\\Users\\vsiripuram\\Desktop\\python\\titanic'
    test_file_path= os.path.join(url,'test.csv')
    train_file_path= os.path.join(url,'train.csv')
    train_df=pd.read_csv(train_file_path,index_col='PassengerId')
    test_df=pd.read_csv(test_file_path,index_col='PassengerId')
    test_df['Survived']=-888
    df=pd.concat((test_df,train_df),axis=0,sort='True')
    return df

def process_data(df):
    # we are using the method chaining concept
    # we are give the ongroup of result set to immedite next group
    return (df
            .assign(Title= lambda x: x.Name.map(get_title))
            #missing values
            .pipe(fill_missing_values)
            .assign(Fare_Bin=lambda x:pd.qcut(x.Fare,4,labels=['very_low','low','high','very_high']))
            .assign(AgeState= lambda x: np.where(x.Age>=18,'Adult','child'))
            .assign(FamilySize=lambda x:x.Parch + x.SibSp + 1)
            .assign(IsMother=lambda x : np.where(((x.Sex=='female') & (x.Parch > 0) & (x.Age > 18) & (x.Title!='Miss')),1,0))
            
            .assign(Cabin=lambda x: np.where(x.Cabin =='T', np.nan , x.Cabin))
            .assign(Deck=lambda x: x.Cabin.map(get_deck))
            
            .assign(IsMale= lambda x : np.where(x.Sex=='male',1,0))
            .pipe(pd.get_dummies,columns=['Deck','Pclass','Title','Fare_Bin','Embarked','AgeState'])
            .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'] , axis=1)
            .pipe(reorder_columns)
           )

def get_title(name):
    title_group={
        'mr':'Mr',
        'mrs' : 'Mrs',
        'miss':'Miss',
        'master':'Master',
        'ms':'Mrs',
        'col':'Officer',
        'rev':'Sir',
        'dr':'Officer',
        'dona':'Lady',
       'don':'Sir',
        'mme':'Mrs',
        'major':'Officer',
        'lady':'Lady',
        'sir':'Sir',
        'mlle':'Miss',
        'capt':'Officer',
       'the countess':'Lady',
        'jonkheer':'Sir'}
    first_name_with_title= name.split(',')[1]
    title=first_name_with_title.split('.')[0]
    title=title.strip().lower()
    return title
    
def fill_missing_values(df):
    df.Embarked.fillna('C',inplace=True)
    median_fare=df[(df.Pclass==3) & (df.Embarked=='S')]['Fare'].median()
    df.Fare.fillna(median_fare,inplace=True)
    return df

def get_deck(Cabin):
    return np.where(pd.notnull(Cabin),str(Cabin)[0].upper(),'z')

def reorder_columns(df):
    columns=[column for column in df.columns if column != 'Survived']
    columns=['Survived']+columns
    df=df[columns]
    return df
def write_data(df):
    url='C:\\Users\\vsiripuram\\Desktop\\python\\titanic'
    write_train_path=os.path.join(url,'train.csv')
    write_test_path=os.path.join(url,'test.csv')
    columns=[column for column in df.columns if column!='Survived']
    df.loc[df.Survived==-888,columns].to_csv(write_test_path)

    
if __name__=='__main__':
    df = read_data()
    df=process_data(df)
    write_data(df)
