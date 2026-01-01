#REQUIRED LIBRARIES=>
#'sys' MODULE IS USED TO EXTRACT THE ERROR DETAILS:
import sys
#TO USE LOGGING IN FUTURE:
import logging

#THIS FUNCTION IS USE TO  CREATE ERROR IN STRUCTURE AND DETAILED MANNER=>
def error_message_details(error, error_detail:sys):
    #sys.exc_info()--> EXTRACT INFO LIKE(error_type, error_value, traceback_object):
    _,_,exc_tb = error_detail.exc_info()
    #EXTRACT FILE NAME FROM TRACEBACK OBJECT:
    file_name = exc_tb.tb_frame.f_code.co_filename
    #CREATES A CUSTOM ERROR MESSAGE:
    error_message = f"ERROR OCCURRED IN PYTHON SCRIPT NAME [{file_name}], LINE NUMBER [{exc_tb.tb_lineno}], ERROR MESSAGE [{str(error)}]"
    #FINAL FORMATTED ERROR MESSAGE RETURN:
    return error_message

#CUSTOM EXCEPTION CLASS (SPECIAL EXCEPTION FOR OUR OWN PROJECT)
class CustomException(Exception):
    #CONSTRUCTOR WHENEVER CUSTOM EXCEPTION WILL BE RAISE THIS WILL BE EXECUTED
    def __init__(self,error_message, error_detail:sys):
        #init CALL OF PARENT EXCEPTION CLASS:
        super().__init__(error_message)
        #CALLING THE FUNCTION TO CREATE FORMATTED ERROR MESSAGE
        self.error_message = error_message_details(error_message, error_detail=error_detail)
    #WHEN WE WILL PRINT EXCEPTION THIS WILL RETURN THE  FORMATTED MESSAGE
    def __str__(self):
        return self.error_message

#NOTE:
'''
1. CONSTRUCTOR NEVER RETURNS ANYTHING
2. sys.exc_info() RETURNS 3 VALUES (error_type, error_value, traceback)
    1.type:(CLASS OF THE ERROR)
        EX: ZeroDivisionError, ValueError, etc
    2.value:(ACTUAL ERROR MESSAGE)
        'division by zero'
        'invalid literal for int()'
    3.traceback:
        1.FILE NAME
        2.LINE NO
        3.WHICH FUNCTION IS RUNNING
'''