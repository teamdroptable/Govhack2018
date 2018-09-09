import pandas as pd
import numpy as np
from sklearn import svm, preprocessing
import pickle


training_data = pd.read_csv('road_crash_data_simple.csv')
#training_data = pd.read_csv('test.csv')


risk_factors = training_data[['Day of the week',
                              'Time of day',
                              'LONGITUDE',
                              'LATITUDE',
                              'Weather',
                              'Lighting',
                              'RoadCondition']].values

risk_factors_processed = preprocessing.scale(risk_factors)


#type_label = np.where(training_data['Type'] == 'Muffin', 0, 1)

type_label = training_data['CrashServerity']




model = svm.SVC(kernel='rbf', C=1, gamma=2**-1, decision_function_shape='ovr')
#small gamma means less complexity whereas gamma = 2** 1 means extremely high (too high)
#model = svm.SVC(kernel='linear', decision_function_shape='ovr')
model.fit(risk_factors_processed, type_label)

pickle_out = open("allFactors.pickle","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

def accident_level(day_of_week, time, long, lat, weather, light, road):
    return model.predict([[day_of_week, time, long, lat, weather, light, road]])


#5,1,149.056890281045,-35.3477908215238,0,0,0
#hopefully returns 0
#print("should be a 0 result:")
#print(accident_level(3,1,149.138901726867,-35.3445942499744,2,2,0))
print('done!')
