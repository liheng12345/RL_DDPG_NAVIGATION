; Auto-generated. Do not edit!


(cl:in-package imitation_learning-srv)


;//! \htmlinclude Mode-request.msg.html

(cl:defclass <Mode-request> (roslisp-msg-protocol:ros-message)
  ((reqmode
    :reader reqmode
    :initarg :reqmode
    :type cl:integer
    :initform 0))
)

(cl:defclass Mode-request (<Mode-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Mode-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Mode-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name imitation_learning-srv:<Mode-request> is deprecated: use imitation_learning-srv:Mode-request instead.")))

(cl:ensure-generic-function 'reqmode-val :lambda-list '(m))
(cl:defmethod reqmode-val ((m <Mode-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader imitation_learning-srv:reqmode-val is deprecated.  Use imitation_learning-srv:reqmode instead.")
  (reqmode m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Mode-request>) ostream)
  "Serializes a message object of type '<Mode-request>"
  (cl:let* ((signed (cl:slot-value msg 'reqmode)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Mode-request>) istream)
  "Deserializes a message object of type '<Mode-request>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'reqmode) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Mode-request>)))
  "Returns string type for a service object of type '<Mode-request>"
  "imitation_learning/ModeRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Mode-request)))
  "Returns string type for a service object of type 'Mode-request"
  "imitation_learning/ModeRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Mode-request>)))
  "Returns md5sum for a message object of type '<Mode-request>"
  "89be0af32cc4dff7129247657bbdf9de")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Mode-request)))
  "Returns md5sum for a message object of type 'Mode-request"
  "89be0af32cc4dff7129247657bbdf9de")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Mode-request>)))
  "Returns full string definition for message of type '<Mode-request>"
  (cl:format cl:nil "int64 reqmode~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Mode-request)))
  "Returns full string definition for message of type 'Mode-request"
  (cl:format cl:nil "int64 reqmode~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Mode-request>))
  (cl:+ 0
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Mode-request>))
  "Converts a ROS message object to a list"
  (cl:list 'Mode-request
    (cl:cons ':reqmode (reqmode msg))
))
;//! \htmlinclude Mode-response.msg.html

(cl:defclass <Mode-response> (roslisp-msg-protocol:ros-message)
  ((setmode
    :reader setmode
    :initarg :setmode
    :type cl:integer
    :initform 0))
)

(cl:defclass Mode-response (<Mode-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Mode-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Mode-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name imitation_learning-srv:<Mode-response> is deprecated: use imitation_learning-srv:Mode-response instead.")))

(cl:ensure-generic-function 'setmode-val :lambda-list '(m))
(cl:defmethod setmode-val ((m <Mode-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader imitation_learning-srv:setmode-val is deprecated.  Use imitation_learning-srv:setmode instead.")
  (setmode m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Mode-response>) ostream)
  "Serializes a message object of type '<Mode-response>"
  (cl:let* ((signed (cl:slot-value msg 'setmode)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Mode-response>) istream)
  "Deserializes a message object of type '<Mode-response>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'setmode) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Mode-response>)))
  "Returns string type for a service object of type '<Mode-response>"
  "imitation_learning/ModeResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Mode-response)))
  "Returns string type for a service object of type 'Mode-response"
  "imitation_learning/ModeResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Mode-response>)))
  "Returns md5sum for a message object of type '<Mode-response>"
  "89be0af32cc4dff7129247657bbdf9de")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Mode-response)))
  "Returns md5sum for a message object of type 'Mode-response"
  "89be0af32cc4dff7129247657bbdf9de")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Mode-response>)))
  "Returns full string definition for message of type '<Mode-response>"
  (cl:format cl:nil "int64 setmode~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Mode-response)))
  "Returns full string definition for message of type 'Mode-response"
  (cl:format cl:nil "int64 setmode~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Mode-response>))
  (cl:+ 0
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Mode-response>))
  "Converts a ROS message object to a list"
  (cl:list 'Mode-response
    (cl:cons ':setmode (setmode msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'Mode)))
  'Mode-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'Mode)))
  'Mode-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Mode)))
  "Returns string type for a service object of type '<Mode>"
  "imitation_learning/Mode")