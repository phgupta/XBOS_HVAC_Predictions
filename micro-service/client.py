import grpc
import pytz
import calendar
import datetime

from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

import HVAC_data_pb2
import HVAC_data_pb2_grpc

_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
HOST_ADDRESS = 'localhost:50058'


def run():

    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel(HOST_ADDRESS) as channel:

        stub = HVAC_data_pb2_grpc.HVACDataStub(channel)

        try:
            d_start = pytz.utc.localize(datetime.datetime(2018, 1, 1, minute=2))
            d_end = d_start + datetime.timedelta(days=2)
            start = int(calendar.timegm(d_start.utctimetuple()) * 1e9)
            end = int(calendar.timegm(d_end.utctimetuple()) * 1e9)

            bldg = "ciee"
            point_type = 'Building_Electric_Meter'

            request = HVAC_data_pb2.HVACRequest(
                building=bldg,
                start=start,
                end=end,
                point_type=point_type,
                Aggregate=HVAC_data_pb2.Aggregate(
                    meter='MEAN',
                    tstat='MAX'
                ),
                Window=HVAC_data_pb2.Window(
                    meter='1m',
                    tstat='1m'
                )
            )

            response = stub.GetHVACData(request)

            for point in response.point:
                print('oat: ', point.oat)

        except grpc.RpcError as e:
            print(e)


if __name__ == '__main__':

    run()
