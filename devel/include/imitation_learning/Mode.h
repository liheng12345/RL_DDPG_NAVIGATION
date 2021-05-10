// Generated by gencpp from file imitation_learning/Mode.msg
// DO NOT EDIT!


#ifndef IMITATION_LEARNING_MESSAGE_MODE_H
#define IMITATION_LEARNING_MESSAGE_MODE_H

#include <ros/service_traits.h>


#include <imitation_learning/ModeRequest.h>
#include <imitation_learning/ModeResponse.h>


namespace imitation_learning
{

struct Mode
{

typedef ModeRequest Request;
typedef ModeResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct Mode
} // namespace imitation_learning


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::imitation_learning::Mode > {
  static const char* value()
  {
    return "89be0af32cc4dff7129247657bbdf9de";
  }

  static const char* value(const ::imitation_learning::Mode&) { return value(); }
};

template<>
struct DataType< ::imitation_learning::Mode > {
  static const char* value()
  {
    return "imitation_learning/Mode";
  }

  static const char* value(const ::imitation_learning::Mode&) { return value(); }
};


// service_traits::MD5Sum< ::imitation_learning::ModeRequest> should match
// service_traits::MD5Sum< ::imitation_learning::Mode >
template<>
struct MD5Sum< ::imitation_learning::ModeRequest>
{
  static const char* value()
  {
    return MD5Sum< ::imitation_learning::Mode >::value();
  }
  static const char* value(const ::imitation_learning::ModeRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::imitation_learning::ModeRequest> should match
// service_traits::DataType< ::imitation_learning::Mode >
template<>
struct DataType< ::imitation_learning::ModeRequest>
{
  static const char* value()
  {
    return DataType< ::imitation_learning::Mode >::value();
  }
  static const char* value(const ::imitation_learning::ModeRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::imitation_learning::ModeResponse> should match
// service_traits::MD5Sum< ::imitation_learning::Mode >
template<>
struct MD5Sum< ::imitation_learning::ModeResponse>
{
  static const char* value()
  {
    return MD5Sum< ::imitation_learning::Mode >::value();
  }
  static const char* value(const ::imitation_learning::ModeResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::imitation_learning::ModeResponse> should match
// service_traits::DataType< ::imitation_learning::Mode >
template<>
struct DataType< ::imitation_learning::ModeResponse>
{
  static const char* value()
  {
    return DataType< ::imitation_learning::Mode >::value();
  }
  static const char* value(const ::imitation_learning::ModeResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // IMITATION_LEARNING_MESSAGE_MODE_H