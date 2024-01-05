
import os

import numpy as np
import win32com.client


class ObjCpCybos:

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=2&page=1&searchString=CpUtil.CpCybos&p=&v=&m=
    """

    obj = win32com.client.Dispatch("CpUtil.CpCybos")

    @classmethod
    def connect(cls, user_id, user_password, cert_password, app_loc=''):
        if not cls.obj.IsConnect():
            os.chdir(app_loc)
            os.system(f"coStarter.exe /prj:cp /id:{user_id} /pwd:{user_password} /pwdcert:{cert_password} /autostart")
            print('Connect Success')
        elif cls.obj.IsConnect():
            print('Already Connected')
        
    @classmethod
    def disconnect(cls):
        cls.obj.PlusDisconnect()
        print('Disconnect Success')

    @classmethod
    def get_remain_count(cls, stock_or_order) -> int:
        """
        stock_or_order: 0:order, 1:stock
        """
        return cls.obj.GetLimitRemainCount(stock_or_order)

    @classmethod
    def get_refresh_time(cls, stock_or_order) -> int:
        """
        stock_or_order: 0:order, 1:stock
        """
        return cls.obj.GetLimitRemainTime(stock_or_order)


class ObjCpTdNew5331A:

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=171&page=1&searchString=CpTdNew5331A&p=&v=&m=
    """

    obj = win32com.client.Dispatch("CpTrade.CpTdNew5331A")

    @classmethod
    def blockrequest(cls):
        cls.obj.BlockRequest()

    @classmethod
    def set_input(cls, input_dict:dict):
        """
        input_dict: {creon_type:value}
        """
        for key,value in input_dict.items():
            cls.obj.SetInputValue(key,value)

    @classmethod
    def get_header(cls, output_dict:dict) -> dict:
        """
        output_dict: {creon_type:name}
        """
        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = cls.obj.GetHeaderValue(key)

        return res_dict
    
# class ObjCpTd5341:

#     """
#     https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=291&seq=174&page=1&searchString=&p=&v=&m=
#     """

#     obj = win32com.client.Dispatch("CpTrade.CpTd5341")

#     @classmethod
#     def blockrequest(cls):
#         cls.obj.BlockRequest()

#     @classmethod
#     def set_input(cls, input_dict:dict):
#         """
#         input_dict: {creon_type:value}
#         """
#         for key,value in input_dict.items():
#             cls.obj.SetInputValue(key,value)

#     @classmethod

    
class ObjCpTd6032:

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=291&seq=264&page=1&searchString=&p=&v=&m=
    """

    obj = win32com.client.Dispatch("CpTrade.CpTd6032")

    @classmethod
    def blockrequest(cls):
        cls.obj.BlockRequest()

    @classmethod
    def set_input(cls, input_dict:dict):
        """
        input_dict: {0-(str):계좌번호, 1-(str):상품관리구분코드}
        """
        for key,value in input_dict.items():
            cls.obj.SetInputValue(key,value)
    
    @classmethod
    def get_header(cls, output_dict:{0:"조회 요청건수",1:"잔량평가손익",2:"매도실현손익",3:"수익률"}) -> dict:
        """
        output_dict: {creon_type:name}
        """
        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = cls.obj.GetHeaderValue(key)

        return res_dict
    
    @classmethod
    def get_data(
            cls,
            output_dict:dict={
                0:"종목명",
                1:"신용일자",
                2:"전일잔고",
                3:"금일매수수량",
                4:"금일매도수량",
                5:"금일잔고",
                6:"평균매입단가",
                7:"평균매도단가",
                8:"현재가",
                9:"잔량평가손익",
                10:"매도실현손익",
                11:"수익율(%)",
                12:"종목코드"
            },
            to_numpy=True
        ) -> dict:
        """
        output_dict: {creon_type:name}
        """
        number_of_data = cls.obj.GetHeaderValue(7)

        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = []

        for key,value in output_dict.items():
            for i in range(number_of_data):
                res_dict[value].append(cls.obj.GetDataValue(key,i))
        
        if to_numpy:
            for key,value in res_dict.items():
                res_dict[key] = np.array(value)

        return res_dict
    

class ObjCpTd6033:

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=176&page=1&searchString=CpTrade.CpTd6033&p=&v=&m=
    """

    obj = win32com.client.Dispatch("CpTrade.CpTd6033")

    @classmethod
    def blockrequest(cls):
        cls.obj.BlockRequest()

    @classmethod
    def set_input(cls, input_dict:dict):
        """
        input_dict: {creon_type:value}
        """
        for key,value in input_dict.items():
            cls.obj.SetInputValue(key,value)

    @classmethod
    def get_header(cls, output_dict:dict) -> dict:
        """
        output_dict: {creon_type:name}
        """
        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = cls.obj.GetHeaderValue(key)

        return res_dict

    @classmethod
    def get_data(cls, output_dict:dict, to_numpy=True) -> dict:
        """
        output_dict: {creon_type:name}
        """
        number_of_data = cls.obj.GetHeaderValue(7)

        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = []

        for key,value in output_dict.items():
            for i in range(number_of_data):
                res_dict[value].append(cls.obj.GetDataValue(key,i))
        
        if to_numpy:
            for key,value in res_dict.items():
                res_dict[key] = np.array(value)

        return res_dict