import joblib
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib

housing = datasets.fetch_california_housing()

x = housing.data
print(x.shape)
y = housing.target

poly = PolynomialFeatures()
x = poly.fit_transform(x)
print(x.shape)
#  Data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=432)


# Model training
#lr = LinearRegression()
#rfr = RandomForestRegressor(n_jobs=5)
gbr = HistGradientBoostingRegressor()


model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate = 0.05
        )
model.fit(x_train, y_train)

# Saving my model
joblib.dump(model, "california_housing.joblib")
y_pred = model.predict(x_test)
# Model evaluation
r2 = r2_score(y_test, y_pred)
print(r2)

