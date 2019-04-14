import grpc
import pandas as pd
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent))

import PowerConsumptionPrediction_pb2
import PowerConsumptionPrediction_pb2_grpc

HOST_ADDRESS = 'localhost:1234'  # CHECK: Change port!


def run():

    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel(HOST_ADDRESS) as channel:

        stub = PowerConsumptionPrediction_pb2_grpc.PowerConsumptionPredictionStub(channel)

        try:

            bldg = "ciee"
            start = 1555198200000000000
            end = 1555212600000000000
            window = '15m'
            dic = {
                'hvac_zone_centralzone': 0
            }

            lst = []
            for key, value in dic.items():
                lst.append(PowerConsumptionPrediction_pb2.Request.Dict(
                    zone=key,
                    state=value
                ))

            # Create gRPC request object
            request = PowerConsumptionPrediction_pb2.Request(
                building=bldg,
                start=start,
                end=end,
                window=window,
                map_zone_state=lst
            )

            response = stub.GetPowerConsumptionPrediction(request)

            df = pd.DataFrame()
            for point in response.point:
                df = df.append([[point.time, point.power]])

            df.columns = ['datetime', 'power']
            df.set_index('datetime', inplace=True)
            df.to_csv(bldg + '.csv')

        except grpc.RpcError as e:
            print(e)


if __name__ == '__main__':

    run()
