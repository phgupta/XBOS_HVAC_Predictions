""" gRPC Server & client examples - https://grpc.io/docs/tutorials/basic/python.html """

import time
import grpc
from concurrent import futures

import sys
sys.path.append('../..')
from Import_Data import Import_Data

import MeterData_pb2
import MeterData_pb2_grpc

# CHECK: Change port
METER_DATA_HOST_ADDRESS = 'localhost:1234'
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class MeterDataServicer(MeterData_pb2_grpc.MeterDataServicer):

    def __init__(self):
        self.import_data_obj = Import_Data()

    def get_data_from_request(self, request):

        site = request.buildings
        start = request.start
        end = request.end
        point_type = request.point_type
        aggregate = request.aggregate
        window = request.window

        # CHECK: Do error checking for all parameters here

        df, map_uuid_meter = self.import_data_obj.get_meter_data(site=site,
                                                                 start=start, end=end, point_type=point_type,
                                                                 agg=aggregate, window=window)
        df.columns = ['power']

        result = []
        for index, row in df.iterrows():
            point = MeterData_pb2.MeterDataPoint(time=str(index), power=row['power'])
            result.append(point)

        return MeterData_pb2.Reply(point=result)

    def GetMeterData(self, request, context):

        result = self.get_data_from_request(request)

        if not result:
            # List of status codes: https: // github.com / grpc / grpc / blob / master / doc / statuscodes.md
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            # context.set_details(error)
            return MeterData_pb2.Reply()
        else:
            return result


if __name__ == '__main__':

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    MeterData_pb2_grpc.add_MeterDataServicer_to_server(MeterDataServicer(), server)
    server.add_insecure_port(METER_DATA_HOST_ADDRESS)
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
