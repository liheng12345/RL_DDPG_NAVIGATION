// Auto-generated. Do not edit!

// (in-package imitation_learning.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class ModeRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.reqmode = null;
    }
    else {
      if (initObj.hasOwnProperty('reqmode')) {
        this.reqmode = initObj.reqmode
      }
      else {
        this.reqmode = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ModeRequest
    // Serialize message field [reqmode]
    bufferOffset = _serializer.int64(obj.reqmode, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ModeRequest
    let len;
    let data = new ModeRequest(null);
    // Deserialize message field [reqmode]
    data.reqmode = _deserializer.int64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'imitation_learning/ModeRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '41ce7d6c57f10377e74d425e7406f66e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int64 reqmode
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ModeRequest(null);
    if (msg.reqmode !== undefined) {
      resolved.reqmode = msg.reqmode;
    }
    else {
      resolved.reqmode = 0
    }

    return resolved;
    }
};

class ModeResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.setmode = null;
    }
    else {
      if (initObj.hasOwnProperty('setmode')) {
        this.setmode = initObj.setmode
      }
      else {
        this.setmode = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ModeResponse
    // Serialize message field [setmode]
    bufferOffset = _serializer.int64(obj.setmode, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ModeResponse
    let len;
    let data = new ModeResponse(null);
    // Deserialize message field [setmode]
    data.setmode = _deserializer.int64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'imitation_learning/ModeResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4b347db3338f4a9756b97090c1da15e6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int64 setmode
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ModeResponse(null);
    if (msg.setmode !== undefined) {
      resolved.setmode = msg.setmode;
    }
    else {
      resolved.setmode = 0
    }

    return resolved;
    }
};

module.exports = {
  Request: ModeRequest,
  Response: ModeResponse,
  md5sum() { return '89be0af32cc4dff7129247657bbdf9de'; },
  datatype() { return 'imitation_learning/Mode'; }
};
