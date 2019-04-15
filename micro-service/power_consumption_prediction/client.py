import grpc
import pandas as pd

import PowerConsumptionPrediction_pb2
import PowerConsumptionPrediction_pb2_grpc

# CHECK: Change port!
HOST_ADDRESS = 'localhost:1234'


def run():

    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel(HOST_ADDRESS) as channel:

        stub = PowerConsumptionPrediction_pb2_grpc.PowerConsumptionPredictionStub(channel)

        try:

            bldg = "ciee"
            start   = 1555376400000000000
            end     = 1555387200000000000
            window = '15T'

            # Specify all zones and their state values.
            dic = {
                'hvac_zone_centralzone': 0,
                'hvac_zone_eastzone': 1,
                'hvac_zone_northzone': 2,
                'hvac_zone_southzone': 3
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
            df.to_csv(bldg + '-predictions.csv')

        except grpc.RpcError as e:
            print(e)


if __name__ == '__main__':

    run()
