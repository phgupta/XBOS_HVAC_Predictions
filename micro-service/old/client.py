import grpc
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

import HVAC_data_pb2
import HVAC_data_pb2_grpc

HOST_ADDRESS = 'localhost:1234'


def run():

    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel(HOST_ADDRESS) as channel:

        stub = HVAC_data_pb2_grpc.HVACDataStub(channel)

        try:

            start = '2018-01-01T00:00:00Z'
            end = '2018-01-15T00:00:00Z'
            bldg = ["ciee"]
            point_type = 'Building_Electric_Meter'

            request = HVAC_data_pb2.HVACRequest(
                buildings=bldg,
                start=start,
                end=end,
                point_type=point_type,
                agg=HVAC_data_pb2.HVACRequest.Aggregate(
                    meter='MEAN',
                    tstat='MAX'
                ),
                window=HVAC_data_pb2.HVACRequest.Window(
                    meter='1m',
                    tstat='1m'
                )
            )

            response = stub.GetHVACData(request)

            df = pd.DataFrame()
            for point in response.point:
                temp = [point.time, point.power, point.oat]
                df = df.append([temp + list(point.iat) + list(point.state)])

            df.to_csv('response.csv')

        except grpc.RpcError as e:
            print(e)


if __name__ == '__main__':

    run()
