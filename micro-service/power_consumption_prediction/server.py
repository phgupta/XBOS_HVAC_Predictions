""" gRPC Server & client examples - https://grpc.io/docs/tutorials/basic/python.html """

import time
import grpc
import pytz
import json
import pickle
import pandas as pd
from concurrent import futures
from datetime import datetime
import xbos_services_getter
import sklearn

import PowerConsumptionPrediction_pb2
import PowerConsumptionPrediction_pb2_grpc

# CHECK: Change port!
HOST_ADDRESS = 'localhost:1234'
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class PowerConsumptionPredictionServicer(PowerConsumptionPrediction_pb2_grpc.PowerConsumptionPredictionServicer):

    def __init__(self):
        """ Constructor. """

        self.model_folder = 'models/'
        self.building_name = None
        self.window = None
        self.start_time = None
        self.end_time = None
        self.map_zone_state = {}
        self.zones = None

        # Currently supported buildings get from microservice
        self.supported_buildings = [
            "orinda-public-library",
            "orinda-community-center",
            "hayward-station-1",
            "hayward-station-8",
            "avenal-animal-shelter",
            "avenal-movie-theatre",
            "avenal-public-works-yard",
            "avenal-recreation-center",
            "avenal-veterans-hall",
            "south-berkeley-senior-center",
            "north-berkeley-senior-center",
            "berkeley-corporation-yard",
            "word-of-faith-cc",
            "local-butcher-shop",
            "jesse-turner-center",
            "ciee",
            "csu-dominguez-hills"
        ]

    @staticmethod
    def add_time_features(data, year=False, month=False, week=False, tod=True, dow=True):
        """

        Parameters
        ----------
        data    : pd.DataFrame()
            Dataframe to add time features to.
        year    : bool
            Year.
        month   : bool
            Month (0-11)
        week    : bool
            Week (0-51)
        tod     : bool
            Time of Day (0-23)
        dow     : bool
            Day of Week (0-6)

        Returns
        -------
        pd.DataFrame()
            Dataframe with time features added as columns.

        """

        var_to_expand = []

        if year:
            data["year"] = data.index.year
            var_to_expand.append("year")
        if month:
            data["month"] = data.index.month
            var_to_expand.append("month")
        if week:
            data["week"] = data.index.week
            var_to_expand.append("week")
        if tod:
            data["tod"] = data.index.hour
            var_to_expand.append("tod")
        if dow:
            data["dow"] = data.index.weekday
            var_to_expand.append("dow")

        # One-hot encode the time features
        for var in var_to_expand:
            add_var = pd.get_dummies(data[var], prefix=var, drop_first=True)

            # Add all the columns to the model data
            data = data.join(add_var)

            # Drop the original column that was expanded
            data.drop(columns=[var], inplace=True)

        return data

    def get_parameters(self, request):
        """ Storing and error checking request parameters.

        Parameters
        ----------
        request     : gRPC request
            Contains parameters to fetch data.

        Returns
        -------
        str
            Error message.

        """

        # CHECK: Move all error first and then do retrieve paramters from request obj.

        # Retrieve parameters from gRPC request object
        self.building_name = request.building
        self.window = request.window
        self.start_time = datetime.utcfromtimestamp(float(request.start/1e9)).replace(tzinfo=pytz.utc)
        self.end_time = datetime.utcfromtimestamp(float(request.end/1e9)).replace(tzinfo=pytz.utc)

        for dic in request.map_zone_state:
            self.map_zone_state[dic.zone] = dic.state

        # List of zones in building
        building_zone_names_stub = xbos_services_getter.get_building_zone_names_stub()
        self.zones = xbos_services_getter.get_zones(building_zone_names_stub, self.building_name)

        if set(self.map_zone_state.keys()) != set(self.zones):
            return "invalid request, specify all zones and their states of the building."

        if any(not elem for elem in [self.building_name, self.window, self.start_time, self.end_time]):
            return "invalid request, empty param(s)"

        # Add error checking for window

        if request.end > int((time.time() + _ONE_DAY_IN_SECONDS * 6) * 1e9):
            return "invalid request, end date is too far in the future, max is 6 days from now"

        if request.start < int(time.time() * 1e9):
            return "invalid request, start date is in the past."

        if request.start >= request.end:
            return "invalid request, start date is equal or after end date."

        if request.building not in self.supported_buildings:
            return "invalid request, building not found, supported buildings:" + str(self.supported_buildings)

        # # Other error checkings
        # duration = utils.get_window_in_sec(request.window)
        # if duration <= 0:
        #     return None, "invalid request, duration is negative or zero"
        # if request.start + (duration * 1e9) > request.end:
        #     return None, "invalid request, start date + window is greater than end date"

    def get_predictions(self):
        """ Retrieve model weights and return predictions.

        Returns
        -------
        gRPC response
            List of points containing the datetime and power consumption prediction.

        """

        indices = pd.date_range(start=self.start_time, freq=self.window, end=self.end_time)
        X_test = pd.DataFrame(index=indices)

        time_features = []
        with open(self.model_folder + self.building_name + '-model.json') as json_file:
            data = json.load(json_file)
            time_features = data['time_features']
            train_zone_cols = data['zone_columns']

        # Add time features to X_test
        cols = []
        for feature in time_features:
            if feature == 'tod':
                for i in range(1, 24):
                    cols.append('tod_' + str(i))
            if feature == 'dow':
                for i in range(1, 7):
                    cols.append('dow_' + str(i))

        X_test = self.add_time_features(X_test)
        X_test_cols = list(X_test)
        for col in cols:
            if col not in X_test_cols:
                X_test[col] = 0

        # Add zone features to X_test
        for col in train_zone_cols:
            X_test[col] = 0

        for zone, state in self.map_zone_state.items():
            for col in train_zone_cols:
                if col.startswith(zone):
                    # CHECK: Hardcoded for now!
                    if int(col[-3]) == state:
                        X_test[col] = 1
                    else:
                        X_test[col] = 0

        loaded_model = pickle.load(open(self.model_folder + self.building_name + '-model.sav', 'rb'))
        y_pred = loaded_model.predict(X_test)

        result = []
        for i in range(len(y_pred)):
            point = PowerConsumptionPrediction_pb2.Reply.PowerConsumptionPredictionPoint(
                time=indices[i].strftime('%Y-%m-%d %H:%M:%S'), power=y_pred[i]
            )
            result.append(point)

        return PowerConsumptionPrediction_pb2.Reply(point=result)

    def GetPowerConsumptionPrediction(self, request, context):
        """

        Parameters
        ----------
        request     : gRPC request
            Contains parameters to fetch data.
        context     : ???
            ???

        Returns
        -------
        gRPC response
            List of points containing the datetime and power consumption.

        """

        error = self.get_parameters(request)

        if error:
            # List of status codes: https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(error)
            return PowerConsumptionPrediction_pb2.Reply()
        else:

            result = self.get_predictions()

            if not result:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                return PowerConsumptionPrediction_pb2.Reply()

        return result


if __name__ == '__main__':

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    PowerConsumptionPrediction_pb2_grpc.add_PowerConsumptionPredictionServicer_to_server(
        PowerConsumptionPredictionServicer(), server
    )
    server.add_insecure_port(HOST_ADDRESS)
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
