import pandas as pd
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('in-vehicle-coupon-recommendation.csv')

# Create design matrices
Y, X = dmatrices('Y ~ 0 + destination + passanger + weather + temperature + time + coupon + \
 expiration + gender + age + maritalStatus + has_children + education + \
 occupation + income + Bar + CoffeeHouse + CarryAway + \
 RestaurantLessThan20 + Restaurant20To50 + toCoupon_GEQ5min + \
 toCoupon_GEQ15min + toCoupon_GEQ25min + direction_same + direction_opp', df, return_type='dataframe')
y = Y['Y'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train Decision Tree
dt_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
dt_model.fit(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
rf_model.fit(X_train, y_train)

# Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Save models
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

# Save the feature names for the app
feature_names = list(X.columns)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("Models trained and saved successfully!")