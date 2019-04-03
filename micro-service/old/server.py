""" gRPC Server & client examples - https://grpc.io/docs/tutorials/basic/python.html """

import time
import grpc
from concurrent import futures

import sys
sys.path.append('..')
from Import_Data import Import_Data

import HVAC_data_pb2
import HVAC_data_pb2_grpc

# CHECK: Need different port for this application
# HVAC_DATA_HOST_ADDRESS = os.environ["OUTDOOR_TEMPERATURE_HISTORICAL_HOST_ADDRESS"]
HVAC_DATA_HOST_ADDRESS = 'localhost:1234'
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


# CHECK: Change class name later to see if it still works
class HVACDataServicer(HVAC_data_pb2_grpc.HVACDataServicer):

    def __init__(self):
        self.import_data_obj = Import_Data()

    def get_data_from_request(self, request):

        site = request.buildings
        start = request.start
        end = request.end
        point_type = request.point_type

        aggregate = {
            'meter': request.agg.meter,
            'tstat': request.agg.tstat
        }

        window = {
            'meter': request.window.meter,
            'tstat': request.window.tstat
        }

        # CHECK: Do error checking for all parameters here

        for site, df in self.import_data_obj.get_data(site, start, end, point_type, aggregate, window):

            result = []
            for index, row in df.iterrows():

                # CHECK: Can be optimized or rewritten later
                states = [int(row[col]) for col in df.columns if col.startswith('s')]
                iat = [int(row[col]) for col in df.columns if col.startswith('t')]

                hvac_point = HVAC_data_pb2.HVACPoint(
                    time=str(index),
                    oat=float(row['oat']),
                    power=float(row['power']),
                    iat=iat,
                    state=states
                )
                result.append(hvac_point)

            return HVAC_data_pb2.HVACReply(point=result)

    def GetHVACData(self, request, context):

        df_hvac_reply = self.get_data_from_request(request)

        if not df_hvac_reply:
            # List of status codes: https: // github.com / grpc / grpc / blob / master / doc / statuscodes.md
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            # context.set_details(error)
            return HVAC_data_pb2.HVACReply()
        else:
            return df_hvac_reply


if __name__ == '__main__':

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    HVAC_data_pb2_grpc.add_HVACDataServicer_to_server(HVACDataServicer(), server)
    server.add_insecure_port(HVAC_DATA_HOST_ADDRESS)
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
