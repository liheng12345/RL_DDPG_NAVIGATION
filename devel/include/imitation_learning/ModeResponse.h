// Generated by gencpp from file imitation_learning/ModeResponse.msg
// DO NOT EDIT!


#ifndef IMITATION_LEARNING_MESSAGE_MODERESPONSE_H
#define IMITATION_LEARNING_MESSAGE_MODERESPONSE_H


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
struct ModeResponse_
{
  typedef ModeResponse_<ContainerAllocator> Type;

  ModeResponse_()
    : setmode(0)  {
    }
  ModeResponse_(const ContainerAllocator& _alloc)
    : setmode(0)  {
  (void)_alloc;
    }



   typedef int64_t _setmode_type;
  _setmode_type setmode;





  typedef boost::shared_ptr< ::imitation_learning::ModeResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::imitation_learning::ModeResponse_<ContainerAllocator> const> ConstPtr;

}; // struct ModeResponse_

typedef ::imitation_learning::ModeResponse_<std::allocator<void> > ModeResponse;

typedef boost::shared_ptr< ::imitation_learning::ModeResponse > ModeResponsePtr;
typedef boost::shared_ptr< ::imitation_learning::ModeResponse const> ModeResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::imitation_learning::ModeResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::imitation_learning::ModeResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::imitation_learning::ModeResponse_<ContainerAllocator1> & lhs, const ::imitation_learning::ModeResponse_<ContainerAllocator2> & rhs)
{
  return lhs.setmode == rhs.setmode;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::imitation_learning::ModeResponse_<ContainerAllocator1> & lhs, const ::imitation_learning::ModeResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace imitation_learning

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::imitation_learning::ModeResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::imitation_learning::ModeResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::imitation_learning::ModeResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::imitation_learning::ModeResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::imitation_learning::ModeResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::imitation_learning::ModeResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::imitation_learning::ModeResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "4b347db3338f4a9756b97090c1da15e6";
  }

  static const char* value(const ::imitation_learning::ModeResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x4b347db3338f4a97ULL;
  static const uint64_t static_value2 = 0x56b97090c1da15e6ULL;
};

template<class ContainerAllocator>
struct DataType< ::imitation_learning::ModeResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "imitation_learning/ModeResponse";
  }

  static const char* value(const ::imitation_learning::ModeResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::imitation_learning::ModeResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "int64 setmode\n"
;
  }

  static const char* value(const ::imitation_learning::ModeResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::imitation_learning::ModeResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.setmode);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ModeResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::imitation_learning::ModeResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::imitation_learning::ModeResponse_<ContainerAllocator>& v)
  {
    s << indent << "setmode: ";
    Printer<int64_t>::stream(s, indent + "  ", v.setmode);
  }
};

} // namespace message_operations
} // namespace ros

#endif // IMITATION_LEARNING_MESSAGE_MODERESPONSE_H
