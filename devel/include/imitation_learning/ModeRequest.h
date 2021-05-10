// Generated by gencpp from file imitation_learning/ModeRequest.msg
// DO NOT EDIT!


#ifndef IMITATION_LEARNING_MESSAGE_MODEREQUEST_H
#define IMITATION_LEARNING_MESSAGE_MODEREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace imitation_learning
{
template <class ContainerAllocator>
struct ModeRequest_
{
  typedef ModeRequest_<ContainerAllocator> Type;

  ModeRequest_()
    : reqmode(0)  {
    }
  ModeRequest_(const ContainerAllocator& _alloc)
    : reqmode(0)  {
  (void)_alloc;
    }



   typedef int64_t _reqmode_type;
  _reqmode_type reqmode;





  typedef boost::shared_ptr< ::imitation_learning::ModeRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::imitation_learning::ModeRequest_<ContainerAllocator> const> ConstPtr;

}; // struct ModeRequest_

typedef ::imitation_learning::ModeRequest_<std::allocator<void> > ModeRequest;

typedef boost::shared_ptr< ::imitation_learning::ModeRequest > ModeRequestPtr;
typedef boost::shared_ptr< ::imitation_learning::ModeRequest const> ModeRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::imitation_learning::ModeRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::imitation_learning::ModeRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::imitation_learning::ModeRequest_<ContainerAllocator1> & lhs, const ::imitation_learning::ModeRequest_<ContainerAllocator2> & rhs)
{
  return lhs.reqmode == rhs.reqmode;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::imitation_learning::ModeRequest_<ContainerAllocator1> & lhs, const ::imitation_learning::ModeRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace imitation_learning

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::imitation_learning::ModeRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::imitation_learning::ModeRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::imitation_learning::ModeRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::imitation_learning::ModeRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::imitation_learning::ModeRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::imitation_learning::ModeRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::imitation_learning::ModeRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "41ce7d6c57f10377e74d425e7406f66e";
  }

  static const char* value(const ::imitation_learning::ModeRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x41ce7d6c57f10377ULL;
  static const uint64_t static_value2 = 0xe74d425e7406f66eULL;
};

template<class ContainerAllocator>
struct DataType< ::imitation_learning::ModeRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "imitation_learning/ModeRequest";
  }

  static const char* value(const ::imitation_learning::ModeRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::imitation_learning::ModeRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "int64 reqmode\n"
;
  }

  static const char* value(const ::imitation_learning::ModeRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::imitation_learning::ModeRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.reqmode);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ModeRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::imitation_learning::ModeRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::imitation_learning::ModeRequest_<ContainerAllocator>& v)
  {
    s << indent << "reqmode: ";
    Printer<int64_t>::stream(s, indent + "  ", v.reqmode);
  }
};

} // namespace message_operations
} // namespace ros

#endif // IMITATION_LEARNING_MESSAGE_MODEREQUEST_H
