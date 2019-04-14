""" gRPC Server & client examples - https://grpc.io/docs/tutorials/basic/python.html """

import time
import grpc
import pytz
import pandas as pd
from concurrent import futures
from datetime import datetime
import xbos_services_getter

# import sys
# sys.path.append('../..')

import PowerConsumptionPrediction_pb2
import PowerConsumptionPrediction_pb2_grpc

HOST_ADDRESS = 'localhost:1234'  # CHECK: Change port!
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class PowerConsumptionPredictionServicer(PowerConsumptionPrediction_pb2_grpc.PowerConsumptionPredictionServicer):

    def __init__(self):
        """ Constructor. Stores raw data, model weights and predictions. """
        self.data = None

        # """ Constructor: Creates an instance of Import_Data which uses pymortar to fetch meter data. """
        # self.import_data_obj = Import_Data()

    def get_data_from_request(self, request):
        """

        Parameters
        ----------
        request     : gRPC request
            Contains parameters to fetch data.

        Returns
        -------
        gRPC response
            List of points containing the datetime and power consumption.

        """

        # Retrieve parameters from gRPC request object
        bldg = request.building
        wndw = request.window
        start_datetime = datetime.utcfromtimestamp(float(request.start/1e9)).replace(tzinfo=pytz.utc)
        end_datetime = datetime.utcfromtimestamp(float(request.end/1e9)).replace(tzinfo=pytz.utc)

        print('start_datetime: ', start_datetime)
        print(('end_datetime: ', end_datetime))

        print('Retrieving data from get_actions_historics()...')

        # Parameters for training data
        start = datetime(2018, 1, 1, 0, 0, 0, 0, pytz.UTC)
        end = datetime(2018, 1, 15, 0, 0, 0, 0, pytz.UTC)
        point_type = 'Building_Electric_Meter'
        agg = 'MEAN'
        window = '15m'

        map_zone_state = {}
        for point in request.map_zone_state:
            map_zone_state[point.zone] = point.state

        df_states = pd.DataFrame()
        indoor_historic_stub = xbos_services_getter.get_indoor_historic_stub()

        for zone in map_zone_state.keys():

            temp = xbos_services_getter.get_actions_historic(indoor_historic_stub, building=bldg,
                                                             start=start, end=end,
                                                             window=window, zone=zone)
            df_states[zone] = temp

        print('Retrieved data from get_actions_historics()!!!')
        print('Retrieving data from get_meter_data()...')

        meter_data_historical_stub = xbos_services_getter.get_meter_data_historical_stub()
        df_meter = xbos_services_getter.get_meter_data_historical(meter_data_stub=meter_data_historical_stub,
                                                                  bldg=[bldg], point_type=point_type,
                                                                  start=start, end=end,
                                                                  aggregate=agg, window=window)
        df_meter.columns = ['power']

        print('Retrieved data from get_meter_data()!!!')

        df_result = df_states.join(df_meter)
        result = []
        for index, row in df_result.iterrows():
            point = PowerConsumptionPrediction_pb2.Reply.PowerConsumptionPredictionPoint(time=str(index), power=row['power'])
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

        result = self.get_data_from_request(request)

        if not result:
            # List of status codes: https: // github.com / grpc / grpc / blob / master / doc / statuscodes.md
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            # context.set_details(error)
            return PowerConsumptionPrediction_pb2.Reply()
        else:
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
