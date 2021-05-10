;; Auto-generated. Do not edit!


(when (boundp 'imitation_learning::Mode)
  (if (not (find-package "IMITATION_LEARNING"))
    (make-package "IMITATION_LEARNING"))
  (shadow 'Mode (find-package "IMITATION_LEARNING")))
(unless (find-package "IMITATION_LEARNING::MODE")
  (make-package "IMITATION_LEARNING::MODE"))
(unless (find-package "IMITATION_LEARNING::MODEREQUEST")
  (make-package "IMITATION_LEARNING::MODEREQUEST"))
(unless (find-package "IMITATION_LEARNING::MODERESPONSE")
  (make-package "IMITATION_LEARNING::MODERESPONSE"))

(in-package "ROS")





(defclass imitation_learning::ModeRequest
  :super ros::object
  :slots (_reqmode ))

(defmethod imitation_learning::ModeRequest
  (:init
   (&key
    ((:reqmode __reqmode) 0)
    )
   (send-super :init)
   (setq _reqmode (round __reqmode))
   self)
  (:reqmode
   (&optional __reqmode)
   (if __reqmode (setq _reqmode __reqmode)) _reqmode)
  (:serialization-length
   ()
   (+
    ;; int64 _reqmode
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int64 _reqmode
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _reqmode (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _reqmode) (= (length (_reqmode . bv)) 2)) ;; bignum
              (write-long (ash (elt (_reqmode . bv) 0) 0) s)
              (write-long (ash (elt (_reqmode . bv) 1) -1) s))
             ((and (class _reqmode) (= (length (_reqmode . bv)) 1)) ;; big1
              (write-long (elt (_reqmode . bv) 0) s)
              (write-long (if (>= _reqmode 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _reqmode s)(write-long (if (>= _reqmode 0) 0 #xffffffff) s)))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int64 _reqmode
#+(or :alpha :irix6 :x86_64)
      (setf _reqmode (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _reqmode (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;;
   self)
  )

(defclass imitation_learning::ModeResponse
  :super ros::object
  :slots (_setmode ))

(defmethod imitation_learning::ModeResponse
  (:init
   (&key
    ((:setmode __setmode) 0)
    )
   (send-super :init)
   (setq _setmode (round __setmode))
   self)
  (:setmode
   (&optional __setmode)
   (if __setmode (setq _setmode __setmode)) _setmode)
  (:serialization-length
   ()
   (+
    ;; int64 _setmode
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int64 _setmode
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _setmode (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _setmode) (= (length (_setmode . bv)) 2)) ;; bignum
              (write-long (ash (elt (_setmode . bv) 0) 0) s)
              (write-long (ash (elt (_setmode . bv) 1) -1) s))
             ((and (class _setmode) (= (length (_setmode . bv)) 1)) ;; big1
              (write-long (elt (_setmode . bv) 0) s)
              (write-long (if (>= _setmode 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _setmode s)(write-long (if (>= _setmode 0) 0 #xffffffff) s)))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int64 _setmode
#+(or :alpha :irix6 :x86_64)
      (setf _setmode (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _setmode (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;;
   self)
  )

(defclass imitation_learning::Mode
  :super ros::object
  :slots ())

(setf (get imitation_learning::Mode :md5sum-) "89be0af32cc4dff7129247657bbdf9de")
(setf (get imitation_learning::Mode :datatype-) "imitation_learning/Mode")
(setf (get imitation_learning::Mode :request) imitation_learning::ModeRequest)
(setf (get imitation_learning::Mode :response) imitation_learning::ModeResponse)

(defmethod imitation_learning::ModeRequest
  (:response () (instance imitation_learning::ModeResponse :init)))

(setf (get imitation_learning::ModeRequest :md5sum-) "89be0af32cc4dff7129247657bbdf9de")
(setf (get imitation_learning::ModeRequest :datatype-) "imitation_learning/ModeRequest")
(setf (get imitation_learning::ModeRequest :definition-)
      "int64 reqmode
---
int64 setmode
")

(setf (get imitation_learning::ModeResponse :md5sum-) "89be0af32cc4dff7129247657bbdf9de")
(setf (get imitation_learning::ModeResponse :datatype-) "imitation_learning/ModeResponse")
(setf (get imitation_learning::ModeResponse :definition-)
      "int64 reqmode
---
int64 setmode
")



(provide :imitation_learning/Mode "89be0af32cc4dff7129247657bbdf9de")


