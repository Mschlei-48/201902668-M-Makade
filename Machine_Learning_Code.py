import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', sep=';')
print(df.head())
#Melt the columns other than  H03,H05, and H16 into two columns sales and date, where sales will have the 
# values of the melted columns, while date will have the column names of the melted columns
df = df.melt(id_vars=['H03', 'H05', 'H16'],
                        var_name="Date", value_name="sales")

#Turn the date column into a string
df['Date'] = df['Date'].astype(str)
# Make a attern for capturing the date fields
pattern = r'(MO)(\d{2})(\d{4})'
print(pattern)
#Extract the pattern and put the pattern field into columns Prefiz, Month and Year respectively
df[['Prefix', 'Month', 'Year']] = df['Date'].str.extract(pattern)
#Concatenating the Month and year into one variable called date
df['date'] = pd.to_datetime(df['Year'].astype(str)+'-' +
                            df['Month'].astype(str)+'-'+'01', format='%Y-%m-%d')

#Drop the clumns we used to get the date column
df.drop(columns=['Date', 'Year', 'Prefix', 'Month'], inplace=True)

#Encode the variables using ordinal encoding
df["H03"] = pd.Categorical(df["H03"], categories=df["H03"].unique(), ordered=True).codes
df["H05"] = pd.Categorical(df["H05"], categories=df["H05"].unique(), ordered=True).codes
df["H16"] = pd.Categorical(df["H16"], categories=df["H16"].unique(), ordered=True).codes
df["date"] = pd.Categorical(df["date"], categories=df["date"].unique(), ordered=True).codes

#Split data into featurs and labels
features=df.loc[:,["H03","H05","H16","date"]]
labels=df.loc[:,"sales"]

#Split data into train and test
x_train,x_test,y_train,y_test=train_test_split(features,labels,
    test_size=0.3,random_state=25)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

#Load the knn model
knn = KNeighborsRegressor(n_neighbors=3)

# Train the model on the training data
knn.fit(x_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(x_test)

#Save the results
r2_score = r2_score(y_test, y_pred)
mse_score=mean_squared_error(y_test,y_pred)
#accuracy = accuracy_score(y_test, y_pred)
print("R2 Score:", r2_score)
print("Mean squared error:",mse_score)
#print("Accuracy:",accuracy)


#plot the actual vs predicted plot
plt.scatter(y_test,y_pred,c="blue",label="Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
# Writing the result to a file
with open('results.txt', 'w') as f:
    f.write(f"R2 Score: {r2_score}\n MSE error: {mse_score}")
#with open('mse_results.txt','w') as f:
    #f.write(f'MSE error: {mse_score}\n')
#save the plot
plt.savefig("actual_vs_predicted.png",dpi=300)