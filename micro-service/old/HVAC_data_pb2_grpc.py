# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import HVAC_data_pb2 as HVAC__data__pb2


class HVACDataStub(object):
  """The HVAC Data definition.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GetHVACData = channel.unary_unary(
        '/hvac_data.HVACData/GetHVACData',
        request_serializer=HVAC__data__pb2.HVACRequest.SerializeToString,
        response_deserializer=HVAC__data__pb2.HVACReply.FromString,
        )


class HVACDataServicer(object):
  """The HVAC Data definition.
  """

  def GetHVACData(self, request, context):
    """A simple RPC.
    CHECK: Sends the outside temperature for a given building, within a duration (start, end), and a requested window
    An error is returned if there is no meter, weather or tstat data for the given request.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_HVACDataServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GetHVACData': grpc.unary_unary_rpc_method_handler(
          servicer.GetHVACData,
          request_deserializer=HVAC__data__pb2.HVACRequest.FromString,
          response_serializer=HVAC__data__pb2.HVACReply.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'hvac_data.HVACData', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))