import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
# Title
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('california Housing Price Prediction')




st.image('https://wallpapercave.com/wp/wp2464233.jpg')

st.header('Model of housing prices to predict median house values in california ',divider=True)

# st.subheader('''User Must Enter Given values to predict Price:
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')


st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://png.pngtree.com/thumb_back/fh260/background/20230804/pngtree-an-upside-graph-showing-prices-and-houses-in-the-market-image_13000262.jpg')

temp_df = pd.read_csv('california.csv')


random.seed(12)

all_values = []


for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('house_priced _pred_ridge_model.pkl','rb')as f:
    chatgpt = pickle.load(f)



price = chatgpt.predict(final_value)[0]  

import time

st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('predicting price')

place = st.empty()
place.image('https://media1.tenor.com/images/e3976785784da43de7a91bf8bd74276c/tenor.gif',width = 2000)

if price>0:

    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    body = f'predicted median house price:${round(price,2)}thousand dollars'
    placeholder.empty()
    place.empty()

    st.success(body)
    
else:
    body = 'invalid house feature values'
    st.warning(body)


    
