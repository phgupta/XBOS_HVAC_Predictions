import grpc
import pandas as pd

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

            bldg = "orinda-community-center"
            start = 1555228800000000000
            end = 1555236000000000000
            window = '15T'
            dic = {
                'hvac_zone_ac_7': 0,
                'hvac_zone_rm7': 0,
                'hvac_zone_kinder_gym': 0,
                'hvac_zone_ac_6': 0,
                'hvac_zone_ac_3': 0,
                'hvac_zone_rm1': 0,
                'hvac_zone_ac_4': 0,
                'hvac_zone_ac_5': 0,
                'hvac_zone_rm6': 0,
                'hvac_zone_ac_1': 0,
                'hvac_zone_front_office': 0,
                'hvac_zone_ac_2': 0,
                'hvac_zone_rm2': 0,
                'hvac_zone_ac_8': 0
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

            print('before')
            response = stub.GetPowerConsumptionPrediction(request)
            print('after')

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
