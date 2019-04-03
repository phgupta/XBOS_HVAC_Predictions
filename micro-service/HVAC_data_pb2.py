# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: HVAC_data.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='HVAC_data.proto',
  package='hvac_data',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0fHVAC_data.proto\x12\thvac_data\"\x81\x02\n\x0bHVACRequest\x12\x11\n\tbuildings\x18\x01 \x03(\t\x12\r\n\x05start\x18\x02 \x01(\t\x12\x0b\n\x03\x65nd\x18\x03 \x01(\t\x12\x12\n\npoint_type\x18\x04 \x01(\t\x12-\n\x03\x61gg\x18\x05 \x01(\x0b\x32 .hvac_data.HVACRequest.Aggregate\x12-\n\x06window\x18\x06 \x01(\x0b\x32\x1d.hvac_data.HVACRequest.Window\x1a)\n\tAggregate\x12\r\n\x05meter\x18\x01 \x01(\t\x12\r\n\x05tstat\x18\x02 \x01(\t\x1a&\n\x06Window\x12\r\n\x05meter\x18\x01 \x01(\t\x12\r\n\x05tstat\x18\x02 \x01(\t\"Q\n\tHVACPoint\x12\x0c\n\x04time\x18\x01 \x01(\t\x12\x0b\n\x03oat\x18\x02 \x01(\x01\x12\r\n\x05power\x18\x03 \x01(\x01\x12\x0b\n\x03iat\x18\x04 \x03(\x01\x12\r\n\x05state\x18\x05 \x03(\x05\"0\n\tHVACReply\x12#\n\x05point\x18\x01 \x03(\x0b\x32\x14.hvac_data.HVACPoint2I\n\x08HVACData\x12=\n\x0bGetHVACData\x12\x16.hvac_data.HVACRequest\x1a\x14.hvac_data.HVACReply\"\x00\x62\x06proto3')
)




_HVACREQUEST_AGGREGATE = _descriptor.Descriptor(
  name='Aggregate',
  full_name='hvac_data.HVACRequest.Aggregate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='meter', full_name='hvac_data.HVACRequest.Aggregate.meter', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tstat', full_name='hvac_data.HVACRequest.Aggregate.tstat', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=207,
  serialized_end=248,
)

_HVACREQUEST_WINDOW = _descriptor.Descriptor(
  name='Window',
  full_name='hvac_data.HVACRequest.Window',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='meter', full_name='hvac_data.HVACRequest.Window.meter', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tstat', full_name='hvac_data.HVACRequest.Window.tstat', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=250,
  serialized_end=288,
)

_HVACREQUEST = _descriptor.Descriptor(
  name='HVACRequest',
  full_name='hvac_data.HVACRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='buildings', full_name='hvac_data.HVACRequest.buildings', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start', full_name='hvac_data.HVACRequest.start', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end', full_name='hvac_data.HVACRequest.end', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='point_type', full_name='hvac_data.HVACRequest.point_type', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='agg', full_name='hvac_data.HVACRequest.agg', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='window', full_name='hvac_data.HVACRequest.window', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_HVACREQUEST_AGGREGATE, _HVACREQUEST_WINDOW, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=288,
)


_HVACPOINT = _descriptor.Descriptor(
  name='HVACPoint',
  full_name='hvac_data.HVACPoint',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='time', full_name='hvac_data.HVACPoint.time', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='oat', full_name='hvac_data.HVACPoint.oat', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='power', full_name='hvac_data.HVACPoint.power', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='iat', full_name='hvac_data.HVACPoint.iat', index=3,
      number=4, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='state', full_name='hvac_data.HVACPoint.state', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=290,
  serialized_end=371,
)


_HVACREPLY = _descriptor.Descriptor(
  name='HVACReply',
  full_name='hvac_data.HVACReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='point', full_name='hvac_data.HVACReply.point', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=373,
  serialized_end=421,
)

_HVACREQUEST_AGGREGATE.containing_type = _HVACREQUEST
_HVACREQUEST_WINDOW.containing_type = _HVACREQUEST
_HVACREQUEST.fields_by_name['agg'].message_type = _HVACREQUEST_AGGREGATE
_HVACREQUEST.fields_by_name['window'].message_type = _HVACREQUEST_WINDOW
_HVACREPLY.fields_by_name['point'].message_type = _HVACPOINT
DESCRIPTOR.message_types_by_name['HVACRequest'] = _HVACREQUEST
DESCRIPTOR.message_types_by_name['HVACPoint'] = _HVACPOINT
DESCRIPTOR.message_types_by_name['HVACReply'] = _HVACREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HVACRequest = _reflection.GeneratedProtocolMessageType('HVACRequest', (_message.Message,), dict(

  Aggregate = _reflection.GeneratedProtocolMessageType('Aggregate', (_message.Message,), dict(
    DESCRIPTOR = _HVACREQUEST_AGGREGATE,
    __module__ = 'HVAC_data_pb2'
    # @@protoc_insertion_point(class_scope:hvac_data.HVACRequest.Aggregate)
    ))
  ,

  Window = _reflection.GeneratedProtocolMessageType('Window', (_message.Message,), dict(
    DESCRIPTOR = _HVACREQUEST_WINDOW,
    __module__ = 'HVAC_data_pb2'
    # @@protoc_insertion_point(class_scope:hvac_data.HVACRequest.Window)
    ))
  ,
  DESCRIPTOR = _HVACREQUEST,
  __module__ = 'HVAC_data_pb2'
  # @@protoc_insertion_point(class_scope:hvac_data.HVACRequest)
  ))
_sym_db.RegisterMessage(HVACRequest)
_sym_db.RegisterMessage(HVACRequest.Aggregate)
_sym_db.RegisterMessage(HVACRequest.Window)

HVACPoint = _reflection.GeneratedProtocolMessageType('HVACPoint', (_message.Message,), dict(
  DESCRIPTOR = _HVACPOINT,
  __module__ = 'HVAC_data_pb2'
  # @@protoc_insertion_point(class_scope:hvac_data.HVACPoint)
  ))
_sym_db.RegisterMessage(HVACPoint)

HVACReply = _reflection.GeneratedProtocolMessageType('HVACReply', (_message.Message,), dict(
  DESCRIPTOR = _HVACREPLY,
  __module__ = 'HVAC_data_pb2'
  # @@protoc_insertion_point(class_scope:hvac_data.HVACReply)
  ))
_sym_db.RegisterMessage(HVACReply)



_HVACDATA = _descriptor.ServiceDescriptor(
  name='HVACData',
  full_name='hvac_data.HVACData',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=423,
  serialized_end=496,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetHVACData',
    full_name='hvac_data.HVACData.GetHVACData',
    index=0,
    containing_service=None,
    input_type=_HVACREQUEST,
    output_type=_HVACREPLY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_HVACDATA)

DESCRIPTOR.services_by_name['HVACData'] = _HVACDATA

# @@protoc_insertion_point(module_scope)
