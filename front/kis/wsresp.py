from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


__all__ = [
    "WSResp005","WSResp012",                      # 체결통보(주식,선옵)
    "WSResp003","WSResp004","WSResp041",          # 국내주식(체결,호가,예상체결)
    "WSResp047","WSResp048","WSResp049",          # 국내주식(회원사,프로그램,운영정보)
    "WSResp010","WSResp011",                      # 지수선물(체결,호가)
    "WSResp014","WSResp015",                      # 지수옵션(체결,호가)
    "WSResp022","WSResp023",                      # 상품선물(체결,호가)
    "WSResp024","WSResp025","WSResp042",          # 국내주식시간외(예상체결,호가,체결)
    "WSResp026","WSResp027","WSResp028",          # 국내지수(체결,예상체결,프로그램)
    "WSResp029","WSResp030","WSResp031",          # 주식선물(체결,호가,예상체결)
    "WSResp032","WSResp033","WSResp034",          # EUREX야간옵션
    "WSResp044","WSResp045","WSResp046",          # 주식옵션(체결,호가,예상체결)
    "WSResp051",                                  # ETF(NAV)
    "WSResp061","WSResp062","WSResp063",          # ELW(체결,호가,예상체결)
    "WSResp064","WSResp065",                      # CME야간선물(체결,호가)
    "WSResp065","WSResp067",                      # 체결통보(CME,EUREX)
    "WSResp003NXT","WSResp004NXT","WSResp048NXT", # 국내주식(체결,호가,프로그램) - NXT
    "WSResp003UNI","WSResp004UNI","WSResp048UNI"  # 국내주식(체결,호가,프로그램) - KRX & NXT 통합
]


@dataclass
class MetaWSResp(metaclass=ABCMeta):
    """웹소켓 응답 메타클래스"""

    _rawbytes: list[bytes]
    "원본 list[bytes]"

    @staticmethod
    @abstractmethod
    def trid(ismock: bool=False) -> str:
        """
        tr_id (하드코딩)

        Args:
            ismock (bool): 모의투자 여부. 기본 False.

        Note:
            실전/모의 tr_id가 동일할 경우 ismock 값 무시
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def len() -> int:
        "원본 list[bytes] 길이(하드코딩)"
        raise NotImplementedError()
    
@dataclass
class WSResp003(MetaWSResp):
    "국내주식 실시간체결가 (KRX)[실시간-003]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0STCNT0"

    @staticmethod
    def len() -> int:
        return 46
    
    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def cntg_hour(self) -> str:
        "체결시간"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def prpr(self) -> int:
        "현재가"
        return int(self._rawbytes[2].decode("utf-8"))
    
    @property
    def prdy_vrss_sign(self) -> str:
        "전일대비부호"
        return self._rawbytes[3].decode("utf-8")
    
    @property
    def prdy_vrss(self) -> int:
        "전일대비"
        return int(self._rawbytes[4].decode("utf-8"))
    
    @property
    def prdy_ctrt(self) -> float:
        "전일대비율"
        return float(self._rawbytes[5].decode("utf-8"))
    
    @property
    def wghn_avrg_prc(self) -> float:
        "가중평균가"
        return float(self._rawbytes[6].decode("utf-8"))

    @property
    def oprc(self) -> int:
        "시가"
        return int(self._rawbytes[7].decode("utf-8"))
    
    @property
    def hgpr(self) -> int:
        "고가"
        return int(self._rawbytes[8].decode("utf-8"))
    
    @property
    def lwpr(self) -> int:
        "저가"
        return int(self._rawbytes[9].decode("utf-8"))
    
    @property
    def askp1(self) -> int:
        "매도호가1"
        return int(self._rawbytes[10].decode("utf-8"))
    
    @property
    def bidp1(self) -> int:
        "매수호가1"
        return int(self._rawbytes[11].decode("utf-8"))
    
    @property
    def cntg_vol(self) -> int:
        "체결거래량"
        return int(self._rawbytes[12].decode("utf-8"))

    @property
    def acml_vol(self) -> int:
        "누적거래량"
        return int(self._rawbytes[13].decode("utf-8"))
    
    @property
    def acml_tr_pbmn(self) -> int:
        "누적거래대금"
        return int(self._rawbytes[14].decode("utf-8"))
    
    @property
    def seln_cntg_csnu(self) -> int:
        "매도체결건수"
        return int(self._rawbytes[15].decode("utf-8"))
    
    @property
    def shnu_cntg_csnu(self) -> int:
        "매수체결건수"
        return int(self._rawbytes[16].decode("utf-8"))
    
    @property
    def ntby_cntg_csnu(self) -> int:
        "순매수체결건수"
        return int(self._rawbytes[17].decode("utf-8"))
    
    @property
    def cttr(self) -> float:
        "체결강도"
        return float(self._rawbytes[18].decode("utf-8"))
    
    @property
    def seln_cntg_smtn(self) -> int:
        "총매도수량"
        return int(self._rawbytes[19].decode("utf-8"))

    @property
    def shnu_cntg_smtn(self) -> int:
        "총매수수량"
        return int(self._rawbytes[20].decode("utf-8"))
    
    @property
    def ccld_dvsn(self) -> str:
        "쳬결구분"
        return self._rawbytes[21].decode("utf-8")

    @property
    def shnu_rate(self) -> float:
        "매수비율"
        return float(self._rawbytes[22].decode("utf-8"))
    
    @property
    def prdy_vol_vrss_acml_vol_rate(self) -> float:
        "전일거래량대비등락율"
        return float(self._rawbytes[23].decode("utf-8"))

    @property
    def oprc_hour(self) -> str:
        "시가시간"
        return self._rawbytes[24].decode("utf-8")

    @property
    def oprc_vrss_sign(self) -> str:
        "시가대비부호"
        return self._rawbytes[25].decode("utf-8")
    
    @property
    def oprc_vrss_prpr(self) -> int:
        "시가대비"
        return int(self._rawbytes[26].decode("utf-8"))

    @property
    def hgpr_hour(self) -> str:
        "고가시간"
        return self._rawbytes[27].decode("utf-8")
    
    @property
    def hgpr_vrss_sign(self) -> str:
        "고가대비부호"
        return self._rawbytes[28].decode("utf-8")
    
    @property
    def hgpr_vrss_prpr(self) -> int:
        "고가대비"
        return int(self._rawbytes[29].decode("utf-8"))
    
    @property
    def lwpr_hour(self) -> str:
        "저가시간"
        return self._rawbytes[30].decode("utf-8")
    
    @property
    def lwpr_vrss_sign(self) -> str:
        "저가대비부호"
        return self._rawbytes[31].decode("utf-8")
    
    @property
    def lwpr_vrss_prpr(self) -> int:
        "저가대비"
        return int(self._rawbytes[32].decode("utf-8"))
    
    @property
    def bsop_date(self) -> str:
        "영업일자"
        return self._rawbytes[33].decode("utf-8")
    
    @property
    def new_mkop_cls_code(self) -> str:
        "(신)장운영구분코드"
        return self._rawbytes[34].decode("utf-8")

    @property
    def trht_yn(self) -> str:
        "거래정지여부"
        return self._rawbytes[35].decode("utf-8")
    
    @property
    def askp_rsqn1(self) -> int:
        "매도호가잔량1"
        return int(self._rawbytes[36].decode("utf-8"))
    
    @property
    def bidp_rsqn1(self) -> int:
        "매수호가잔량1"
        return int(self._rawbytes[37].decode("utf-8"))
    
    @property
    def total_askp_rsqn(self) -> int:
        "총매도호가잔량"
        return int(self._rawbytes[38].decode("utf-8"))
    
    @property
    def total_bidp_rsqn(self) -> int:
        "총매수호가잔량"
        return int(self._rawbytes[39].decode("utf-8"))
    
    @property
    def vol_tnrt(self) -> float:
        "거래량회전율"
        return float(self._rawbytes[40].decode("utf-8"))

    @property
    def prdy_smns_hour_acml_vol(self) -> int:
        "전일동시간누적거래량"
        return int(self._rawbytes[41].decode("utf-8"))

    @property
    def prdy_smns_hour_acml_vol_rate(self) -> float:
        "전일동시간누적거래량대비율"
        return float(self._rawbytes[42].decode("utf-8"))
    
    @property
    def hour_cls_code(self) -> str:
        "시간구분코드"
        return self._rawbytes[43].decode("utf-8")
    
    @property
    def mrkt_trtm_cls_code(self) -> str:
        "임의종료구분코드"
        return self._rawbytes[44].decode("utf-8")

    @property
    def vi_stnd_prc(self) -> int:
        "VI기준가"
        return int(self._rawbytes[45].decode("utf-8"))

@dataclass
class WSResp004(MetaWSResp):
    "국내주식 실시간호가 (KRX)[실시간-004]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0STASP0"

    @staticmethod
    def len() -> int:
        return 59
    
    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def bsop_hour(self) -> str:
        "영업시간"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def hour_cls_code(self) -> str:
        "시간구분코드"
        return self._rawbytes[2].decode("utf-8")
    
    @property
    def askp1(self) -> int:
        "매도호가1"
        return int(self._rawbytes[3].decode("utf-8"))
    
    @property
    def askp2(self) -> int:
        "매도호가2"
        return int(self._rawbytes[4].decode("utf-8"))
    
    @property
    def askp3(self) -> int:
        "매도호가3"
        return int(self._rawbytes[5].decode("utf-8"))
    
    @property
    def askp4(self) -> int:
        "매도호가4"
        return int(self._rawbytes[6].decode("utf-8"))
    
    @property
    def askp5(self) -> int:
        "매도호가5"
        return int(self._rawbytes[7].decode("utf-8"))
    
    @property
    def askp6(self) -> int:
        "매도호가6"
        return int(self._rawbytes[8].decode("utf-8"))
    
    @property
    def askp7(self) -> int:
        "매도호가7"
        return int(self._rawbytes[9].decode("utf-8"))
    
    @property
    def askp8(self) -> int:
        "매도호가8"
        return int(self._rawbytes[10].decode("utf-8"))
    
    @property
    def askp9(self) -> int:
        "매도호가9"
        return int(self._rawbytes[11].decode("utf-8"))
    
    @property
    def askp10(self) -> int:
        "매도호가10"
        return int(self._rawbytes[12].decode("utf-8"))
    
    @property
    def bidp1(self) -> int:
        "매수호가1"
        return int(self._rawbytes[13].decode("utf-8"))
    
    @property
    def bidp2(self) -> int:
        "매수호가2"
        return int(self._rawbytes[14].decode("utf-8"))
    
    @property
    def bidp3(self) -> int:
        "매수호가3"
        return int(self._rawbytes[15].decode("utf-8"))
    
    @property
    def bidp4(self) -> int:
        "매수호가4"
        return int(self._rawbytes[16].decode("utf-8"))
    
    @property
    def bidp5(self) -> int:
        "매수호가5"
        return int(self._rawbytes[17].decode("utf-8"))
    
    @property
    def bidp6(self) -> int:
        "매수호가6"
        return int(self._rawbytes[18].decode("utf-8"))
    
    @property
    def bidp7(self) -> int:
        "매수호가7"
        return int(self._rawbytes[19].decode("utf-8"))
    
    @property
    def bidp8(self) -> int:
        "매수호가8"
        return int(self._rawbytes[20].decode("utf-8"))
    
    @property
    def bidp9(self) -> int:
        "매수호가9"
        return int(self._rawbytes[21].decode("utf-8"))
    
    @property
    def bidp10(self) -> int:
        "매수호가10"
        return int(self._rawbytes[22].decode("utf-8"))
    
    @property
    def askp_rsqn1(self) -> int:
        "매도호가잔량1"
        return int(self._rawbytes[23].decode("utf-8"))
    
    @property
    def askp_rsqn2(self) -> int:
        "매도호가잔량2"
        return int(self._rawbytes[24].decode("utf-8"))
    
    @property
    def askp_rsqn3(self) -> int:
        "매도호가잔량3"
        return int(self._rawbytes[25].decode("utf-8"))
    
    @property
    def askp_rsqn4(self) -> int:
        "매도호가잔량4"
        return int(self._rawbytes[26].decode("utf-8"))
    
    @property
    def askp_rsqn5(self) -> int:
        "매도호가잔량5"
        return int(self._rawbytes[27].decode("utf-8"))
    
    @property
    def askp_rsqn6(self) -> int:
        "매도호가잔량6"
        return int(self._rawbytes[28].decode("utf-8"))
    
    @property
    def askp_rsqn7(self) -> int:
        "매도호가잔량7"
        return int(self._rawbytes[29].decode("utf-8"))
    
    @property
    def askp_rsqn8(self) -> int:
        "매도호가잔량8"
        return int(self._rawbytes[30].decode("utf-8"))
    
    @property
    def askp_rsqn9(self) -> int:
        "매도호가잔량9"
        return int(self._rawbytes[31].decode("utf-8"))
    
    @property
    def askp_rsqn10(self) -> int:
        "매도호가잔량10"
        return int(self._rawbytes[32].decode("utf-8"))
    
    @property
    def bidp_rsqn1(self) -> int:
        "매수호가잔량1"
        return int(self._rawbytes[33].decode("utf-8"))
    
    @property
    def bidp_rsqn2(self) -> int:
        "매수호가잔량2"
        return int(self._rawbytes[34].decode("utf-8"))
    
    @property
    def bidp_rsqn3(self) -> int:
        "매수호가잔량3"
        return int(self._rawbytes[35].decode("utf-8"))
    
    @property
    def bidp_rsqn4(self) -> int:
        "매수호가잔량4"
        return int(self._rawbytes[36].decode("utf-8"))
    
    @property
    def bidp_rsqn5(self) -> int:
        "매수호가잔량5"
        return int(self._rawbytes[37].decode("utf-8"))
    
    @property
    def bidp_rsqn6(self) -> int:
        "매수호가잔량6"
        return int(self._rawbytes[38].decode("utf-8"))
    
    @property
    def bidp_rsqn7(self) -> int:
        "매수호가잔량7"
        return int(self._rawbytes[39].decode("utf-8"))
    
    @property
    def bidp_rsqn8(self) -> int:
        "매수호가잔량8"
        return int(self._rawbytes[40].decode("utf-8"))
    
    @property
    def bidp_rsqn9(self) -> int:
        "매수호가잔량9"
        return int(self._rawbytes[41].decode("utf-8"))
    
    @property
    def bidp_rsqn10(self) -> int:
        "매수호가잔량10"
        return int(self._rawbytes[42].decode("utf-8"))
    
    @property
    def total_askp_rsqn(self) -> int:
        "총매도호가잔량"
        return int(self._rawbytes[43].decode("utf-8"))
    
    @property
    def total_bidp_rsqn(self) -> int:
        "총매수호가잔량"
        return int(self._rawbytes[44].decode("utf-8"))
    
    @property
    def ovtm_total_askp_rsqn(self) -> int:
        "시간외총매도호가잔량"
        return int(self._rawbytes[45].decode("utf-8"))
    
    @property
    def ovtm_total_bidp_rsqn(self) -> int:
        "시간외총매수호가잔량"
        return int(self._rawbytes[46].decode("utf-8"))
    
    @property
    def antc_cnpr(self) -> int:
        "예상체결가"
        return int(self._rawbytes[47].decode("utf-8"))
    
    @property
    def antc_cnqn(self) -> int:
        "예상체결량"
        return int(self._rawbytes[48].decode("utf-8"))
    
    @property
    def antc_vol(self) -> int:
        "예상거래량"
        return int(self._rawbytes[49].decode("utf-8"))
    
    @property
    def antc_cntg_vrss(self) -> int:
        "예상체결대비"
        return int(self._rawbytes[50].decode("utf-8"))
    
    @property
    def antc_cntg_vrss_sign(self) -> str:
        "예상체결대비부호"
        return self._rawbytes[51].decode("utf-8")

    @property
    def antc_cntg_prdy_ctrt(self) -> float:
        "예상체결전일대비율"
        return float(self._rawbytes[52].decode("utf-8"))

    @property
    def acml_vol(self) -> int:
        "누적거래량"
        return int(self._rawbytes[53].decode("utf-8"))
    
    @property
    def total_askp_rsqn_icdc(self) -> int:
        "총매도호가잔량증감"
        return int(self._rawbytes[54].decode("utf-8"))
    
    @property
    def total_bidp_rsqn_icdc(self) -> int:
        "총매수호가잔량증감"
        return int(self._rawbytes[55].decode("utf-8"))
    
    @property
    def ovtm_total_askp_icdc(self) -> int:
        "시간외총매도호가잔량증감"
        return int(self._rawbytes[56].decode("utf-8"))
    
    @property
    def ovtm_total_bidp_icdc(self) -> int:
        "시간외총매수호가잔량증감"
        return int(self._rawbytes[57].decode("utf-8"))

@dataclass
class WSResp005(MetaWSResp):
    "국내주식 실시간체결통보[실시간-005]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        if ismock:
            return "H0STCNI9"
        return "H0STCNI0"

    @staticmethod
    def len() -> int:
        return 26

    @property
    def cust_id(self) -> str:
        "고객 ID"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def acnt_no(self) -> str:
        "계좌번호"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def oder_no(self) -> str:
        "주문번호"
        return self._rawbytes[2].decode("utf-8")
    
    @property
    def ooder_no(self) -> str:
        "원주문번호"
        return self._rawbytes[3].decode("utf-8")
    
    @property
    def seln_byov_cls(self) -> str:
        "매도매수구분"
        return self._rawbytes[4].decode("utf-8")

    @property
    def rctf_cls(self) -> str:
        "정정구분"
        return self._rawbytes[5].decode("utf-8")
    
    @property
    def oder_kind(self) -> str:
        "주문종류"
        return self._rawbytes[6].decode("utf-8")
    
    @property
    def oder_cond(self) -> str:
        "주문조건"
        return self._rawbytes[7].decode("utf-8")
    
    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[8].decode("utf-8")
    
    @property
    def cntg_qty(self) -> int:
        "체결수량"
        return int(self._rawbytes[9].decode("utf-8"))
    
    @property
    def cntg_unpr(self) -> int:
        "체결단가"
        return int(self._rawbytes[10].decode("utf-8"))
    
    @property
    def cntg_hour(self) -> str:
        "체결시간"
        return self._rawbytes[11].decode("utf-8")
    
    @property
    def rfus_yn(self) -> str:
        "거부여부"
        return self._rawbytes[12].decode("utf-8")
    
    @property
    def cntg_yn(self) -> str:
        "체결여부"
        return self._rawbytes[13].decode("utf-8")
    
    @property
    def acpt_yn(self) -> str:
        "접수여부"
        return self._rawbytes[14].decode("utf-8")
    
    @property
    def brnc_no(self) -> str:
        "지점번호"
        return self._rawbytes[15].decode("utf-8")
    
    @property
    def oder_qty(self) -> int:
        "주문수량"
        return int(self._rawbytes[16].decode("utf-8"))
    
    @property
    def acnt_name(self) -> str:
        "계좌명"
        return self._rawbytes[17].decode("utf-8")
    
    @property
    def ord_cond_prc(self) -> int:
        "호가조건가격"
        return int(self._rawbytes[18].decode("utf-8"))

    @property
    def ord_exg_gb(self) -> int:
        "거래소구분"
        return int(self._rawbytes[19].decode("utf-8"))
    
    @property
    def popup_yn(self) -> str:
        "실시간체결창 표시여부"
        return self._rawbytes[20].decode("utf-8")
    
    @property
    def filler(self) -> str:
        "필러"
        return self._rawbytes[21].decode("utf-8")
    
    @property
    def crdt_cls(self) -> str:
        "신용구분"
        return self._rawbytes[22].decode("utf-8")
    
    @property
    def crdt_loan_date(self) -> str:
        "신용대출일"
        return self._rawbytes[23].decode("utf-8")
    
    @property
    def cntg_isnm40(self) -> str:
        "체결종목명40"
        return self._rawbytes[24].decode("utf-8")
    
    @property
    def oder_prc(self) -> int:
        "주문가격"
        return int(self._rawbytes[25].decode("utf-8"))

@dataclass
class WSResp010(MetaWSResp):
    "지수선물 실시간체결가[실시간-010]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0IFCNT0"

    @staticmethod
    def len() -> int:
        return 50

    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def bsop_hour(self) -> str:
        "영업시간"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def prdy_vrss(self) -> int:
        "전일대비"
        return int(self._rawbytes[2].decode("utf-8"))
    
    @property
    def prdy_vrss_sign(self) -> str:
        "전일대비부호"
        return self._rawbytes[3].decode("utf-8")
    
    @property
    def prdy_ctrt(self) -> float:
        "전일대비율"
        return float(self._rawbytes[4].decode("utf-8"))
    
    @property
    def prpr(self) -> float:
        "현재가"
        return float(self._rawbytes[5].decode("utf-8"))
    
    @property
    def oprc(self) -> float:
        "시가"
        return float(self._rawbytes[6].decode("utf-8"))
    
    @property
    def hgpr(self) -> float:
        "고가"
        return float(self._rawbytes[7].decode("utf-8"))
    
    @property
    def lwpr(self) -> float:
        "저가"
        return float(self._rawbytes[8].decode("utf-8"))
    
    @property
    def last_cnqn(self) -> int:
        "최종거래량(체결량)"
        return int(self._rawbytes[9].decode("utf-8"))
    
    @property
    def acml_vol(self) -> int:
        "누적거래량"
        return int(self._rawbytes[10].decode("utf-8"))
    
    @property
    def acml_tr_pbmn(self) -> int:
        "누적거래대금"
        return int(self._rawbytes[11].decode("utf-8"))
    
    @property
    def hts_thpr(self) -> float:
        "HTS이론가"
        return float(self._rawbytes[12].decode("utf-8"))
    
    @property
    def mrkt_basis(self) -> float:
        "시장베이시스"
        return float(self._rawbytes[13].decode("utf-8"))
    
    @property
    def dprt(self) -> float:
        "괴리율"
        return float(self._rawbytes[14].decode("utf-8"))

    @property
    def nmsc_fctn_stpl_prc(self) -> float:
        "근월물약정가"
        return float(self._rawbytes[15].decode("utf-8"))

    @property
    def fmsc_fctn_stpl_prc(self) -> float:
        "원월물약정가"
        return float(self._rawbytes[16].decode("utf-8"))
    
    @property
    def spread_prc(self) -> float:
        "스프레드"
        return float(self._rawbytes[17].decode("utf-8"))
    
    @property
    def hts_otst_stpl_qty(self) -> int:
        "HTS미결약정수량"
        return int(self._rawbytes[18].decode("utf-8"))
    
    @property
    def otst_stpl_qty_icdc(self) -> int:
        "미결약정수량증감"
        return int(self._rawbytes[19].decode("utf-8"))
    
    @property
    def oprc_hour(self) -> str:
        "시가시간"
        return self._rawbytes[20].decode("utf-8")
    
    @property
    def oprc_vrss_sign(self) -> str:
        "시가대비부호"
        return self._rawbytes[21].decode("utf-8")
    
    @property
    def oprc_vrss_prpr(self) -> float:
        "시가대비"
        return float(self._rawbytes[22].decode("utf-8"))
    
    @property
    def hgpr_hour(self) -> str:
        "고가시간"
        return self._rawbytes[23].decode("utf-8")
    
    @property
    def hgpr_vrss_sign(self) -> str:
        "고가대비부호"
        return self._rawbytes[24].decode("utf-8")
    
    @property
    def hgpr_vrss_nmix_prpr(self) -> float:
        "고가대비"
        return float(self._rawbytes[25].decode("utf-8"))
    
    @property
    def lwpr_hour(self) -> str:
        "저가시간"
        return self._rawbytes[26].decode("utf-8")
    
    @property
    def lwpr_vrss_sign(self) -> str:
        "저가대비부호"
        return self._rawbytes[27].decode("utf-8")
    
    @property
    def lwpr_vrss_nmix_prpr(self) -> float:
        "저가대비"
        return float(self._rawbytes[28].decode("utf-8"))
    
    @property
    def shnu_rate(self) -> float:
        "매수비율"
        return float(self._rawbytes[29].decode("utf-8"))
    
    @property
    def cttr(self) -> float:
        "체결강도"
        return float(self._rawbytes[30].decode("utf-8"))
    
    @property
    def esdg(self) -> float:
        "괴리도"
        return float(self._rawbytes[31].decode("utf-8"))
    
    @property
    def otst_stpl_rgbf_qty_icdc(self) -> int:
        "미결약정직전수량증감"
        return int(self._rawbytes[32].decode("utf-8"))
    
    @property
    def thpr_basis(self) -> float:
        "이론베이시스"
        return float(self._rawbytes[33].decode("utf-8"))
    
    @property
    def askp1(self) -> float:
        "매도호가1"
        return float(self._rawbytes[34].decode("utf-8"))
    
    @property
    def bidp1(self) -> float:
        "매수호가1"
        return float(self._rawbytes[35].decode("utf-8"))
    
    @property
    def askp_rsqn1(self) -> int:
        "매도호가잔량1"
        return int(self._rawbytes[36].decode("utf-8"))
    
    @property
    def bidp_rsqn1(self) -> int:
        "매수호가잔량1"
        return int(self._rawbytes[37].decode("utf-8"))
    
    @property
    def seln_cntg_csnu(self) -> int:
        "매도체결건수"
        return int(self._rawbytes[38].decode("utf-8"))
    
    @property
    def shnu_cntg_csnu(self) -> int:
        "매수체결건수"
        return int(self._rawbytes[39].decode("utf-8"))
    
    @property
    def ntby_cntg_csnu(self) -> int:
        "순매수체결건수"
        return int(self._rawbytes[40].decode("utf-8"))
    
    @property
    def seln_cntg_smtb(self) -> int:
        "총매도수량"
        return int(self._rawbytes[41].decode("utf-8"))
    
    @property
    def shnu_cntg_smtb(self) -> int:
        "총매수수량"
        return int(self._rawbytes[42].decode("utf-8"))
    
    @property
    def total_askp_rsqn(self) -> int:
        "총매도호가잔량"
        return int(self._rawbytes[43].decode("utf-8"))
    
    @property
    def total_bidp_rsqn(self) -> int:
        "총매수호가잔량"
        return int(self._rawbytes[44].decode("utf-8"))
    
    @property
    def prdy_vol_vrss_acml_vol_rate(self) -> float:
        "전일거래량대비등락율"
        return float(self._rawbytes[45].decode("utf-8"))
    
    @property
    def dscs_bltr_acml_qty(self) -> int:
        "협의대량거래량"
        return int(self._rawbytes[46].decode("utf-8"))
    
    @property
    def dynm_mxpr(self) -> float:
        "실시간상한가"
        return float(self._rawbytes[47].decode("utf-8"))
    
    @property
    def dynm_llam(self) -> float:
        "실시간하한가"
        return float(self._rawbytes[48].decode("utf-8"))
    
    @property
    def dynm_prc_limt_yn(self) -> str:
        "실시간가격제한구분"
        return self._rawbytes[49].decode("utf-8")

@dataclass
class WSResp011(MetaWSResp):
    "지수선물 실시간호가[실시간-011]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0IFASP0"

    @staticmethod
    def len() -> int:
        return 38

    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def bsop_hour(self) -> str:
        "영업시간"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def askp1(self) -> float:
        "매도호가1"
        return float(self._rawbytes[2].decode("utf-8"))
    
    @property
    def askp2(self) -> float:
        "매도호가2"
        return float(self._rawbytes[3].decode("utf-8"))
    
    @property
    def askp3(self) -> float:
        "매도호가3"
        return float(self._rawbytes[4].decode("utf-8"))
    
    @property
    def askp4(self) -> float:
        "매도호가4"
        return float(self._rawbytes[5].decode("utf-8"))
    
    @property
    def askp5(self) -> float:
        "매도호가5"
        return float(self._rawbytes[6].decode("utf-8"))

    @property
    def bidp1(self) -> float:
        "매수호가1"
        return float(self._rawbytes[7].decode("utf-8"))
    
    @property
    def bidp2(self) -> float:
        "매수호가2"
        return float(self._rawbytes[8].decode("utf-8"))
    
    @property
    def bidp3(self) -> float:
        "매수호가3"
        return float(self._rawbytes[9].decode("utf-8"))
    
    @property
    def bidp4(self) -> float:
        "매수호가4"
        return float(self._rawbytes[10].decode("utf-8"))
    
    @property
    def bidp5(self) -> float:
        "매수호가5"
        return float(self._rawbytes[11].decode("utf-8"))
    
    @property
    def askp_csnu1(self) -> int:
        "매도호가건수1"
        return int(self._rawbytes[12].decode("utf-8"))

    @property
    def askp_csnu2(self) -> int:
        "매도호가건수2"
        return int(self._rawbytes[13].decode("utf-8"))
    
    @property
    def askp_csnu3(self) -> int:
        "매도호가건수3"
        return int(self._rawbytes[14].decode("utf-8"))
    
    @property
    def askp_csnu4(self) -> int:
        "매도호가건수4"
        return int(self._rawbytes[15].decode("utf-8"))
    
    @property
    def askp_csnu5(self) -> int:
        "매도호가건수5"
        return int(self._rawbytes[16].decode("utf-8"))
    
    @property
    def bidp_csnu1(self) -> int:
        "매수호가건수1"
        return int(self._rawbytes[17].decode("utf-8"))
    
    @property
    def bidp_csnu2(self) -> int:
        "매수호가건수2"
        return int(self._rawbytes[18].decode("utf-8"))
    
    @property
    def bidp_csnu3(self) -> int:
        "매수호가건수3"
        return int(self._rawbytes[19].decode("utf-8"))
    
    @property
    def bidp_csnu4(self) -> int:
        "매수호가건수4"
        return int(self._rawbytes[20].decode("utf-8"))
    
    @property
    def bidp_csnu5(self) -> int:
        "매수호가건수5"
        return int(self._rawbytes[21].decode("utf-8"))
    
    @property
    def askp_rsqn1(self) -> int:
        "매도호가잔량1"
        return int(self._rawbytes[22].decode("utf-8"))
    
    @property
    def askp_rsqn2(self) -> int:
        "매도호가잔량2"
        return int(self._rawbytes[23].decode("utf-8"))
    
    @property
    def askp_rsqn3(self) -> int:
        "매도호가잔량3"
        return int(self._rawbytes[24].decode("utf-8"))
    
    @property
    def askp_rsqn4(self) -> int:
        "매도호가잔량4"
        return int(self._rawbytes[25].decode("utf-8"))
    
    @property
    def askp_rsqn5(self) -> int:
        "매도호가잔량5"
        return int(self._rawbytes[26].decode("utf-8"))
    
    @property
    def bidp_rsqn1(self) -> int:
        "매수호가잔량1"
        return int(self._rawbytes[27].decode("utf-8"))
    
    @property
    def bidp_rsqn2(self) -> int:
        "매수호가잔량2"
        return int(self._rawbytes[28].decode("utf-8"))
    
    @property
    def bidp_rsqn3(self) -> int:
        "매수호가잔량3"
        return int(self._rawbytes[29].decode("utf-8"))
    
    @property
    def bidp_rsqn4(self) -> int:
        "매수호가잔량4"
        return int(self._rawbytes[30].decode("utf-8"))
    
    @property
    def bidp_rsqn5(self) -> int:
        "매수호가잔량5"
        return int(self._rawbytes[31].decode("utf-8"))
    
    @property
    def total_askp_csnu(self) -> int:
        "총매도호가건수"
        return int(self._rawbytes[32].decode("utf-8"))
    
    @property
    def total_bidp_csnu(self) -> int:
        "총매수호가건수"
        return int(self._rawbytes[33].decode("utf-8"))
    
    @property
    def total_askp_rsqn(self) -> int:
        "총매도호가잔량"
        return int(self._rawbytes[34].decode("utf-8"))
    
    @property
    def total_bidp_rsqn(self) -> int:
        "총매수호가잔량"
        return int(self._rawbytes[35].decode("utf-8"))
    
    @property
    def total_askp_rsqn_icdc(self) -> int:
        "총매도호가잔량증감"
        return int(self._rawbytes[36].decode("utf-8"))
    
    @property
    def total_bidp_rsqn_icdc(self) -> int:
        "총매수호가잔량증감"
        return int(self._rawbytes[37].decode("utf-8"))

@dataclass
class WSResp012(MetaWSResp):
    "선물옵션 실시간체결통보[실시간-012]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        if ismock:
            return "H0IFCNI9"
        return "H0IFCNI0"

    @staticmethod
    def len() -> int:
        return 22
    
    @property
    def cust_id(self) -> str:
        "고객 ID"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def acnt_no(self) -> str:
        "계좌번호"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def oder_no(self) -> str:
        "주문번호"
        return self._rawbytes[2].decode("utf-8")
    
    @property
    def ooder_no(self) -> str:
        "원주문번호"
        return self._rawbytes[3].decode("utf-8")
    
    @property
    def seln_byov_cls(self) -> str:
        "매도매수구분"
        return self._rawbytes[4].decode("utf-8")
    
    @property
    def rctf_cls(self) -> str:
        "정정구분"
        return self._rawbytes[5].decode("utf-8")
    
    @property
    def oder_kind(self) -> str:
        "주문종류"
        return self._rawbytes[6].decode("utf-8")
    
    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[7].decode("utf-8")
    
    @property
    def cntg_qty(self) -> int:
        "체결수량"
        return int(self._rawbytes[8].decode("utf-8"))
    
    @property
    def cntg_unpr(self) -> float:
        "체결단가"
        return float(self._rawbytes[9].decode("utf-8"))
    
    @property
    def cntg_hour(self) -> str:
        "체결시간"
        return self._rawbytes[10].decode("utf-8")
    
    @property
    def rfus_yn(self) -> str:
        "거부여부"
        return self._rawbytes[11].decode("utf-8")
    
    @property
    def cntg_yn(self) -> str:
        "체결여부"
        return self._rawbytes[12].decode("utf-8")
    
    @property
    def acpt_yn(self) -> str:
        "접수여부"
        return self._rawbytes[13].decode("utf-8")
    
    @property
    def brnc_no(self) -> str:
        "지점번호"
        return self._rawbytes[14].decode("utf-8")
    
    @property
    def oder_qty(self) -> int:
        "주문수량"
        return int(self._rawbytes[15].decode("utf-8"))
    
    @property
    def acnt_name(self) -> str:
        "계좌명"
        return self._rawbytes[16].decode("utf-8")
    
    @property
    def cntg_isnm(self) -> str:
        "체결종목명"
        return self._rawbytes[17].decode("utf-8")
    
    @property
    def oder_cond(self) -> str:
        "주문조건"
        return self._rawbytes[18].decode("utf-8")
    
    @property
    def ord_grp(self) -> str:
        "주문그룹ID"
        return self._rawbytes[19].decode("utf-8")

    @property
    def ord_grpseq(self) -> str:
        "주문그룹SEQ"
        return self._rawbytes[20].decode("utf-8")
    
    @property
    def order_prc(self) -> float:
        "주문가격"
        return float(self._rawbytes[21].decode("utf-8"))

@dataclass
class WSResp014(MetaWSResp):
    "지수옵션 실시간체결가[실시간-014]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0IOCNT0"
    
    @staticmethod
    def len() -> int:
        return 58
    
    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def bsop_hour(self) -> str:
        "영업시간"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def prpr(self) -> float:
        "현재가"
        return float(self._rawbytes[2].decode("utf-8"))
    
    @property
    def prdy_vrss_sign(self) -> str:
        "전일대비부호"
        return self._rawbytes[3].decode("utf-8")
    
    @property
    def prdy_vrss(self) -> int:
        "전일대비"
        return int(self._rawbytes[4].decode("utf-8"))
    
    @property
    def prdy_ctrt(self) -> float:
        "전일대비율"
        return float(self._rawbytes[5].decode("utf-8"))
    
    @property
    def oprc(self) -> float:
        "시가"
        return float(self._rawbytes[6].decode("utf-8"))
    
    @property
    def hgpr(self) -> float:
        "고가"
        return float(self._rawbytes[7].decode("utf-8"))
    
    @property
    def lwpr(self) -> float:
        "저가"
        return float(self._rawbytes[8].decode("utf-8"))
    
    @property
    def last_cnqn(self) -> int:
        "최종거래량(체결량)"
        return int(self._rawbytes[9].decode("utf-8"))
    
    @property
    def acml_vol(self) -> int:
        "누적거래량"
        return int(self._rawbytes[10].decode("utf-8"))
    
    @property
    def acml_tr_pbmn(self) -> int:
        "누적거래대금"
        return int(self._rawbytes[11].decode("utf-8"))
    
    @property
    def hts_thpr(self) -> float:
        "HTS이론가"
        return float(self._rawbytes[12].decode("utf-8"))
    
    @property
    def hts_otst_stpl_qty(self) -> int:
        "HTS미결약정수량"
        return int(self._rawbytes[13].decode("utf-8"))
    
    @property
    def otst_stpl_qty_icdc(self) -> int:
        "미결약정수량증감"
        return int(self._rawbytes[14].decode("utf-8"))
    
    @property
    def oprc_hour(self) -> str:
        "시가시간"
        return self._rawbytes[15].decode("utf-8")
    
    @property
    def oprc_vrss_sign(self) -> str:
        "시가대비부호"
        return self._rawbytes[16].decode("utf-8")
    
    @property
    def oprc_vrss_prpr(self) -> float:
        "시가대비"
        return float(self._rawbytes[17].decode("utf-8"))
    
    @property
    def hgpr_hour(self) -> str:
        "고가시간"
        return self._rawbytes[18].decode("utf-8")
    
    @property
    def hgpr_vrss_sign(self) -> str:
        "고가대비부호"
        return self._rawbytes[19].decode("utf-8")
    
    @property
    def hgpr_vrss_prpr(self) -> float:
        "고가대비"
        return float(self._rawbytes[20].decode("utf-8"))
    
    @property
    def lwpr_hour(self) -> str:
        "저가시간"
        return self._rawbytes[21].decode("utf-8")
    
    @property
    def lwpr_vrss_sign(self) -> str:
        "저가대비부호"
        return self._rawbytes[22].decode("utf-8")
    
    @property
    def lwpr_vrss_prpr(self) -> float:
        "저가대비"
        return float(self._rawbytes[23].decode("utf-8"))
    
    @property
    def shnu_rate(self) -> float:
        "매수비율"
        return float(self._rawbytes[24].decode("utf-8"))
    
    @property
    def prmm_val(self) -> float:
        "프리미엄값"
        return float(self._rawbytes[25].decode("utf-8"))
    
    @property
    def invl_val(self) -> float:
        "내재가치값"
        return float(self._rawbytes[26].decode("utf-8"))

    @property
    def tmvl_val(self) -> float:
        "시간가치값"
        return float(self._rawbytes[27].decode("utf-8"))
    
    @property
    def delta(self) -> float:
        "Greeks-델타"
        return float(self._rawbytes[28].decode("utf-8"))
    
    @property
    def gamma(self) -> float:
        "Greeks-감마"
        return float(self._rawbytes[29].decode("utf-8"))
    
    @property
    def vega(self) -> float:
        "Greeks-베가"
        return float(self._rawbytes[30].decode("utf-8"))
    
    @property
    def theta(self) -> float:
        "Greeks-세타"
        return float(self._rawbytes[31].decode("utf-8"))
    
    @property
    def rho(self) -> float:
        "Greeks-로"
        return float(self._rawbytes[32].decode("utf-8"))
    
    @property
    def hts_ints_vltl(self) -> float:
        "HTS내재변동성"
        return float(self._rawbytes[33].decode("utf-8"))
    
    @property
    def esdg(self) -> float:
        "괴리도"
        return float(self._rawbytes[34].decode("utf-8"))
    
    @property
    def otst_stpl_rgbf_qty_icdc(self) -> int:
        "미결약정직전수량증감"
        return int(self._rawbytes[35].decode("utf-8"))
    
    @property
    def thpr_basis(self) -> float:
        "이론베이시스"
        return float(self._rawbytes[36].decode("utf-8"))
    
    @property
    def unas_hist_vltl(self) -> float:
        "역사적변동성"
        return float(self._rawbytes[37].decode("utf-8"))

    @property
    def cttr(self) -> float:
        "체결강도"
        return float(self._rawbytes[38].decode("utf-8"))
    
    @property
    def dprt(self) -> float:
        "괴리율"
        return float(self._rawbytes[39].decode("utf-8"))
    
    @property
    def mrkt_basis(self) -> float:
        "시장베이시스"
        return float(self._rawbytes[40].decode("utf-8"))
    
    @property
    def askp1(self) -> float:
        "매도호가1"
        return float(self._rawbytes[41].decode("utf-8"))
    
    @property
    def bidp1(self) -> float:
        "매수호가1"
        return float(self._rawbytes[42].decode("utf-8"))
    
    @property
    def askp_rsqn1(self) -> int:
        "매도호가잔량1"
        return int(self._rawbytes[43].decode("utf-8"))
    
    @property
    def bidp_rsqn1(self) -> int:
        "매수호가잔량1"
        return int(self._rawbytes[44].decode("utf-8"))
    
    @property
    def seln_cntg_csnu(self) -> int:
        "매도체결건수"
        return int(self._rawbytes[45].decode("utf-8"))
    
    @property
    def shnu_cntg_csnu(self) -> int:
        "매수체결건수"
        return int(self._rawbytes[46].decode("utf-8"))
    
    @property
    def ntby_cntg_csnu(self) -> int:
        "순매수체결건수"
        return int(self._rawbytes[47].decode("utf-8"))
    
    @property
    def seln_cntg_smtb(self) -> int:
        "총매도수량"
        return int(self._rawbytes[48].decode("utf-8"))
    
    @property
    def shnu_cntg_smtb(self) -> int:
        "총매수수량"
        return int(self._rawbytes[49].decode("utf-8"))
    
    @property
    def total_askp_rsqn(self) -> int:
        "총매도호가잔량"
        return int(self._rawbytes[50].decode("utf-8"))
    
    @property
    def total_bidp_rsqn(self) -> int:
        "총매수호가잔량"
        return int(self._rawbytes[51].decode("utf-8"))
    
    @property
    def prdy_vol_vrss_acml_vol_rate(self) -> float:
        "전일거래량대비등락율"
        return float(self._rawbytes[52].decode("utf-8"))
    
    @property
    def avrg_vltl(self) -> float:
        "평균변동성"
        return float(self._rawbytes[53].decode("utf-8"))
    
    @property
    def dscs_lrqn_vol(self) -> int:
        "협의대량누적거래량"
        return int(self._rawbytes[54].decode("utf-8"))

    @property
    def dynm_mxpr(self) -> float:
        "실시간상한가"
        return float(self._rawbytes[55].decode("utf-8"))
    
    @property
    def dynm_llam(self) -> float:
        "실시간하한가"
        return float(self._rawbytes[56].decode("utf-8"))
    
    @property
    def dynm_prc_limt_yn(self) -> str:
        "실시간가격제한구분"
        return self._rawbytes[57].decode("utf-8")

@dataclass
class WSResp015(WSResp011):
    "지수옵션 실시간호가[실시간-015]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0IOASP0"

@dataclass
class WSResp022(WSResp010):
    "상품선물 실시간체결가[실시간-022]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0CFCNT0"
    
@dataclass
class WSResp023(WSResp011):
    "상품선물 실시간호가[실시간-023]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0CFASP0"

@dataclass
class WSResp024(WSResp003):
    "국내주식 시간외 실시간예상체결 [실시간-024]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0STOAC0"
    
    @staticmethod
    def len() -> int:
        return 43
    
    @property
    def hour_cls_code(self) -> str:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def mrkt_trtm_cls_code(self) -> str:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()

    @property
    def vi_stnd_prc(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()

@dataclass
class WSResp025(WSResp004):
    "국내주식 시간외 실시간호가 [실시간-025]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0STOAA0"

@dataclass
class WSResp026(MetaWSResp):
    "국내지수 실시간체결 [실시간-026]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0UPCNT0"
    
    @staticmethod
    def len() -> int:
        return 30
    
    @property
    def bstp_cls_code(self) -> str:
        "업종구분코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def bsop_hour(self) -> str:
        "영업시간"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def prpr(self) -> float:
        "현재가"
        return float(self._rawbytes[2].decode("utf-8"))
    
    @property
    def prdy_vrss_sign(self) -> str:
        "전일대비부호"
        return self._rawbytes[3].decode("utf-8")
    
    @property
    def prdy_vrss(self) -> int:
        "전일대비"
        return int(self._rawbytes[4].decode("utf-8"))
    
    @property
    def acml_vol(self) -> int:
        "누적거래량"
        return int(self._rawbytes[5].decode("utf-8"))
    
    @property
    def acml_tr_pbmn(self) -> int:
        "누적거래대금"
        return int(self._rawbytes[6].decode("utf-8"))
    
    @property
    def pcas_vol(self) -> int:
        "건별거래량"
        return int(self._rawbytes[7].decode("utf-8"))

    @property
    def pcas_tr_pbmn(self) -> int:
        "건별거래대금"
        return int(self._rawbytes[8].decode("utf-8"))
    
    @property
    def prdy_ctrt(self) -> float:
        "전일대비율"
        return float(self._rawbytes[9].decode("utf-8"))
    
    @property
    def oprc(self) -> float:
        "시가"
        return float(self._rawbytes[10].decode("utf-8"))
    
    @property
    def hgpr(self) -> float:
        "고가"
        return float(self._rawbytes[11].decode("utf-8"))
    
    @property
    def lwpr(self) -> float:
        "저가"
        return float(self._rawbytes[12].decode("utf-8"))
    
    @property
    def oprc_vrss_prpr(self) -> float:
        "시가대비"
        return float(self._rawbytes[13].decode("utf-8"))
    
    @property
    def oprc_vrss_sign(self) -> str:
        "시가대비부호"
        return self._rawbytes[14].decode("utf-8")

    @property
    def hgpr_vrss_prpr(self) -> float:
        "고가대비"
        return float(self._rawbytes[15].decode("utf-8"))
    
    @property
    def hgpr_vrss_sign(self) -> str:
        "고가대비부호"
        return self._rawbytes[16].decode("utf-8")
    
    @property
    def lwpr_vrss_prpr(self) -> float:
        "저가대비"
        return float(self._rawbytes[17].decode("utf-8"))
    
    @property
    def lwpr_vrss_sign(self) -> str:
        "저가대비부호"
        return self._rawbytes[18].decode("utf-8")
    
    @property
    def prdy_clpr_vrss_oprc_rate(self) -> float:
        "전일종가대비시가율"
        return float(self._rawbytes[19].decode("utf-8"))
    
    @property
    def prdy_clpr_vrss_hgpr_rate(self) -> float:
        "전일종가대비고가율"
        return float(self._rawbytes[20].decode("utf-8"))
    
    @property
    def prdy_clpr_vrss_lwpr_rate(self) -> float:
        "전일종가대비저가율"
        return float(self._rawbytes[21].decode("utf-8"))
    
    @property
    def uplm_issu_cnt(self) -> int:
        "상한종목수"
        return int(self._rawbytes[22].decode("utf-8"))

    @property
    def ascn_issu_cnt(self) -> int:
        "상승종목수"
        return int(self._rawbytes[23].decode("utf-8"))
    
    @property
    def stnr_issu_cnt(self) -> int:
        "보합종목수"
        return int(self._rawbytes[24].decode("utf-8"))
    
    @property
    def down_issu_cnt(self) -> int:
        "하락종목수"
        return int(self._rawbytes[25].decode("utf-8"))
    
    @property
    def lslm_issu_cnt(self) -> int:
        "하한종목수"
        return int(self._rawbytes[26].decode("utf-8"))
    
    @property
    def qtqt_ascn_issu_cnt(self) -> int:
        "기세상승종목수"
        return int(self._rawbytes[27].decode("utf-8"))

    @property
    def qtqt_down_issu_cnt(self) -> int:
        "기세하락종목수"
        return int(self._rawbytes[28].decode("utf-8"))
    
    @property
    def tick_vrss(self) -> float:
        "틱대비"
        return float(self._rawbytes[29].decode("utf-8"))
    
@dataclass
class WSResp027(WSResp026):
    "국내지수 실시간예상체결 [실시간-027]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0UPANC0"
    
@dataclass
class WSResp028(MetaWSResp):
    "국내지수 실시간프로그램매매 [실시간-028]"

    @staticmethod
    def trid(ismock = False):
        return "H0UPPGM0"
    
    @staticmethod
    def len():
        return 88
    
    @property
    def bstp_cls_code(self) -> str:
        "업종구분코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def bsop_hour(self) -> str:
        "영업시간"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def arbt_seln_entm_cnqn(self) -> int:
        "차익매도위탁체결량"
        return int(self._rawbytes[2].decode("utf-8"))
    
    @property
    def arbt_seln_onsl_cnqn(self) -> int:
        "차익매도자기체결량"
        return int(self._rawbytes[3].decode("utf-8"))
    
    @property
    def arbt_shnu_entm_cnqn(self) -> int:
        "차익매수위탁체결량"
        return int(self._rawbytes[4].decode("utf-8"))
    
    @property
    def arbt_shnu_onsl_cnqn(self) -> int:
        "차익매수자기체결량"
        return int(self._rawbytes[5].decode("utf-8"))
    
    @property
    def nabt_seln_entm_cnqn(self) -> int:
        "비차익매도위탁체결량"
        return int(self._rawbytes[6].decode("utf-8"))
    
    @property
    def nabt_seln_onsl_cnqn(self) -> int:
        "비차익매도자기체결량"
        return int(self._rawbytes[7].decode("utf-8"))
    
    @property
    def nabt_shun_entm_cnqn(self) -> int:
        "비차익매수위탁체결량"
        return int(self._rawbytes[8].decode("utf-8"))
    
    @property
    def nabt_shnu_onsl_cnqn(self) -> int:
        "비차익매수자기체결량"
        return int(self._rawbytes[9].decode("utf-8"))
    
    @property
    def arbt_seln_entm_cntg_amt(self) -> int:
        "차익매도위탁체결금액"
        return int(self._rawbytes[10].decode("utf-8"))
    
    @property
    def arbt_seln_onsl_cntg_amt(self) -> int:
        "차익매도자기체결금액"
        return int(self._rawbytes[11].decode("utf-8"))
    
    @property
    def arbt_shnu_entm_cntg_amt(self) -> int:
        "차익매수위탁체결금액"
        return int(self._rawbytes[12].decode("utf-8"))
    
    @property
    def arbt_shnu_onsl_cntg_amt(self) -> int:
        "차익매수자기체결금액"
        return int(self._rawbytes[13].decode("utf-8"))
    
    @property
    def nabt_seln_entm_cntg_amt(self)  -> int:
        "비차익매도위탁체결금액"
        return int(self._rawbytes[14].decode("utf-8"))
    
    @property
    def nabt_seln_onsl_cntg_amt(self) -> int:
        "비차익매도자기체결금액"
        return int(self._rawbytes[15].decode("utf-8"))
    
    @property
    def nabt_shnu_entm_cntg_amt(self) -> int:
        "비차익매수위탁체결금액"
        return int(self._rawbytes[16].decode("utf-8"))
    
    @property
    def nabt_shnu_onsl_cntg_amt(self) -> int:
        "비차익매수자기체결금액"
        return int(self._rawbytes[17].decode("utf-8"))
    
    @property
    def arbt_smtn_seln_vol(self) -> int:
        "차익매도거래량"
        return int(self._rawbytes[18].decode("utf-8"))

    @property
    def arbt_smtn_seln_vol_rate(self) -> float:
        "차익매도거래량비율"
        return float(self._rawbytes[19].decode("utf-8"))
    
    @property
    def arbt_smtn_seln_tr_pbmn(self) -> int:
        "차익매도거래대금"
        return int(self._rawbytes[20].decode("utf-8"))
    
    @property
    def arbt_smtn_seln_tr_pbmn_rate(self) -> float:
        "차익매도거래대금비율"
        return float(self._rawbytes[21].decode("utf-8"))
    
    @property
    def arbt_smtn_shnu_vol(self) -> int:
        "차익매수거래량"
        return int(self._rawbytes[22].decode("utf-8"))
    
    @property
    def arbt_smtn_shnu_vol_rate(self) -> float:
        "차익매수거래량비율"
        return float(self._rawbytes[23].decode("utf-8"))
    
    @property
    def arbt_smtn_shnu_tr_pbmn(self) -> int:
        "차익매수거래대금"
        return int(self._rawbytes[24].decode("utf-8"))
    
    @property
    def arbt_smtn_shnu_tr_pbmn_rate(self) -> float:
        "차익매수거래대금비율"
        return float(self._rawbytes[25].decode("utf-8"))
    
    @property
    def arbt_smtn_ntby_qty(self) -> int:
        "차익순매수량"
        return int(self._rawbytes[26].decode("utf-8"))
    
    @property
    def arbt_smtn_ntby_qty_rate(self) -> float:
        "차익순매수비율"
        return float(self._rawbytes[27].decode("utf-8"))
    
    @property
    def arbt_smtn_ntby_tr_pbmn(self) -> int:
        "차익순매수대금"
        return int(self._rawbytes[28].decode("utf-8"))
    
    @property
    def arbt_smtn_ntby_tr_pbmn_rate(self) -> float:
        "차익순매수대금비율"
        return float(self._rawbytes[29].decode("utf-8"))
    
    @property
    def nabt_smtn_seln_vol(self) -> int:
        "비차익매도거래량"
        return int(self._rawbytes[30].decode("utf-8"))

    @property
    def nabt_smtn_seln_vol_rate(self) -> float:
        "비차익매도거래량비율"
        return float(self._rawbytes[31].decode("utf-8"))
    
    @property
    def nabt_smtn_seln_tr_pbmn(self) -> int:
        "비차익매도거래대금"
        return int(self._rawbytes[32].decode("utf-8"))
    
    @property
    def nabt_smtn_seln_tr_pbmn_rate(self) -> float:
        "비차익매도거래대금비율"
        return float(self._rawbytes[33].decode("utf-8"))
    
    @property
    def nabt_smtn_shnu_vol(self) -> int:
        "비차익매수거래량"
        return int(self._rawbytes[34].decode("utf-8"))
    
    @property
    def nabt_smtn_shnu_vol_rate(self) -> float:
        "비차익매수거래량비율"
        return float(self._rawbytes[35].decode("utf-8"))
    
    @property
    def nabt_smtn_shnu_tr_pbmn(self) -> int:
        "비차익매수거래대금"
        return int(self._rawbytes[36].decode("utf-8"))
    
    @property
    def nabt_smtn_shnu_tr_pbmn_rate(self) -> float:
        "비차익매수거래대금비율"
        return float(self._rawbytes[37].decode("utf-8"))
    
    @property
    def nabt_smtn_ntby_qty(self) -> int:
        "비차익순매수량"
        return int(self._rawbytes[38].decode("utf-8"))
    
    @property
    def nabt_smtn_ntby_qty_rate(self) -> float:
        "비차익순매수비율"
        return float(self._rawbytes[39].decode("utf-8"))
    
    @property
    def nabt_smtn_ntby_tr_pbmn(self) -> int:
        "비차익순매수대금"
        return int(self._rawbytes[40].decode("utf-8"))
    
    @property
    def nabt_smtn_ntby_tr_pbmn_rate(self) -> float:
        "비차익순매수대금비율"
        return float(self._rawbytes[41].decode("utf-8"))
    
    @property
    def whol_entm_seln_vol(self) -> int:
        "전체위탁매도거래량"
        return int(self._rawbytes[42].decode("utf-8"))
    
    @property
    def entm_seln_vol_rate(self) -> float:
        "위탁매도거래량비율"
        return float(self._rawbytes[43].decode("utf-8"))
    
    @property
    def whol_entm_seln_tr_pbmn(self) -> int:
        "전체위탁매도거래대금"
        return int(self._rawbytes[44].decode("utf-8"))
    
    @property
    def entm_seln_tr_pbmn_rate(self) -> float:
        "위탁매도거래대금비율"
        return float(self._rawbytes[45].decode("utf-8"))
    
    @property
    def whol_shnu_entm_vol(self) -> int:
        "전체위탁매수거래량"
        return int(self._rawbytes[46].decode("utf-8"))
    
    @property
    def entm_shnu_vol_rate(self) -> float:
        "위탁매수거래량비율"
        return float(self._rawbytes[47].decode("utf-8"))
    
    @property
    def whol_shnu_entm_tr_pbmn(self) -> int:
        "전체위탁매수거래대금"
        return int(self._rawbytes[48].decode("utf-8"))
    
    @property
    def entm_shnu_tr_pbmn_rate(self) -> float:
        "위탁매수거래대금비율"
        return float(self._rawbytes[49].decode("utf-8"))
    
    @property
    def whol_entm_ntby_qty(self) -> int:
        "전체위탁순매수량"
        return int(self._rawbytes[50].decode("utf-8"))
    
    @property
    def entm_ntby_qty_rate(self) -> float:
        "위탁순매수량비율"
        return float(self._rawbytes[51].decode("utf-8"))
    
    @property
    def whol_entm_ntby_tr_pbmn(self) -> int:
        "전체위탁순매수대금"
        return int(self._rawbytes[52].decode("utf-8"))
    
    @property
    def entm_ntby_tr_pbmn_rate(self) -> float:
        "위탁순매수대금비율"
        return float(self._rawbytes[53].decode("utf-8"))
    
    @property
    def whol_onsl_seln_vol(self) -> int:
        "전체자기매도거래량"
        return int(self._rawbytes[54].decode("utf-8"))
    
    @property
    def onsl_seln_vol_rate(self) -> float:
        "자기매도거래량비율"
        return float(self._rawbytes[55].decode("utf-8"))
    
    @property
    def whol_onsl_seln_tr_pbmn(self) -> int:
        "전체자기매도거래대금"
        return int(self._rawbytes[56].decode("utf-8"))
    
    @property
    def onsl_seln_tr_pbmn_rate(self) -> float:
        "자기매도거래대금비율"
        return float(self._rawbytes[57].decode("utf-8"))
    
    @property
    def whol_shnu_onsl_vol(self) -> int:
        "전체자기매수거래량"
        return int(self._rawbytes[58].decode("utf-8"))
    
    @property
    def onsl_shnu_vol_rate(self) -> float:
        "자기매수거래량비율"
        return float(self._rawbytes[59].decode("utf-8"))
    
    @property
    def whol_shnu_onsl_tr_pbmn(self) -> int:
        "전체자기매수거래대금"
        return int(self._rawbytes[60].decode("utf-8"))
    
    @property
    def onsl_shnu_tr_pbmn_rate(self) -> float:
        "자기매수거래대금비율"
        return float(self._rawbytes[61].decode("utf-8"))
    
    @property
    def whol_onsl_ntby_qty(self) -> int:
        "전체자기순매수량"
        return int(self._rawbytes[62].decode("utf-8"))
    
    @property
    def onsl_ntby_qty_rate(self) -> float:
        "자기순매수량비율"
        return float(self._rawbytes[63].decode("utf-8"))
    
    @property
    def whol_onsl_ntby_tr_pbmn(self) -> int:
        "전체자기순매수대금"
        return int(self._rawbytes[64].decode("utf-8"))
    
    @property
    def onsl_ntby_tr_pbmn_rate(self) -> float:
        "자기순매수대금비율"
        return float(self._rawbytes[65].decode("utf-8"))
    
    @property
    def total_seln_qty(self) -> int:
        "총매도수량"
        return int(self._rawbytes[66].decode("utf-8"))
    
    @property
    def whol_seln_vol_rate(self) -> float:
        "전체매도거래량비율"
        return float(self._rawbytes[67].decode("utf-8"))

    @property
    def total_seln_tr_pbmn(self) -> int:
        "총매도거래대금"
        return int(self._rawbytes[68].decode("utf-8"))
    
    @property
    def whol_seln_tr_pbmn_rate(self) -> float:
        "전체매도거래대금비율"
        return float(self._rawbytes[69].decode("utf-8"))
    
    @property
    def shnu_cntg_smtn(self) -> int:
        "총매수수량"
        return int(self._rawbytes[70].decode("utf-8"))

    @property
    def whol_shun_vol_rate(self) -> float:
        "전체매수거래량비율"
        return float(self._rawbytes[71].decode("utf-8"))
    
    @property
    def total_shnu_tr_pbmn(self) -> int:
        "총매수거래대금"
        return int(self._rawbytes[72].decode("utf-8"))
    
    @property
    def whol_shnu_tr_pbmn_rate(self) -> float:
        "전체매수거래대금비율"
        return float(self._rawbytes[73].decode("utf-8"))
    
    @property
    def whol_ntby_qty(self) -> int:
        "전체순매수량"
        return int(self._rawbytes[74].decode("utf-8"))
    
    @property
    def whol_smtm_ntby_qty_rate(self) -> float:
        "전체순매수량비율"
        return float(self._rawbytes[75].decode("utf-8"))
    
    @property
    def whol_ntby_tr_pbmn(self) -> int:
        "전체순매수대금"
        return int(self._rawbytes[76].decode("utf-8"))
    
    @property
    def whol_ntby_tr_pbmn_rate(self) -> float:
        "전체순매수대금비율"
        return float(self._rawbytes[77].decode("utf-8"))
    
    @property
    def arbt_entm_ntby_qty(self) -> int:
        "차익위탁순매수량"
        return int(self._rawbytes[78].decode("utf-8"))
    
    @property
    def arbt_entm_ntby_tr_pbmn(self) -> int:
        "차익위탁순매수대금"
        return int(self._rawbytes[79].decode("utf-8"))
    
    @property
    def arby_onsl_ntby_qty(self) -> int:
        "차익자기순매수량"
        return int(self._rawbytes[80].decode("utf-8"))
    
    @property
    def arbt_onsl_ntby_tr_pbmn(self) -> int:
        "차익자기순매수대금"
        return int(self._rawbytes[81].decode("utf-8"))
    
    @property
    def nabt_entm_ntby_qty(self) -> int:
        "비차익위탁순매수량"
        return int(self._rawbytes[82].decode("utf-8"))
    
    @property
    def nabt_entm_ntby_tr_pbmn(self) -> int:
        "비차익위탁순매수대금"
        return int(self._rawbytes[83].decode("utf-8"))
    
    @property
    def nabt_onsl_ntby_qty(self) -> int:
        "비차익자기순매수량"
        return int(self._rawbytes[84].decode("utf-8"))
    
    @property
    def nabt_onsl_ntby_tr_pbmn(self) -> int:
        "비차익자기순매수대금"
        return int(self._rawbytes[85].decode("utf-8"))
    
    @property
    def acml_vol(self) -> int:
        "누적거래량"
        return int(self._rawbytes[86].decode("utf-8"))
    
    @property
    def acml_tr_pbmn(self) -> int:
        "누적거래대금"
        return int(self._rawbytes[87].decode("utf-8"))
    
@dataclass
class WSResp029(WSResp010):
    "주식선물 실시간체결가[실시간-029]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0ZFCNT0"

    @staticmethod
    def len() -> int:
        return 49
    
    @property
    def prdy_vol_vrss_acml_vol_rate(self) -> float:
        "전일거래량대비등락율"
        return float(self._rawbytes[45].decode("utf-8"))
    
    @property
    def dscs_bltr_acml_qty(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def dynm_mxpr(self) -> float:
        "실시간상한가"
        return float(self._rawbytes[46].decode("utf-8"))
    
    @property
    def dynm_llam(self) -> float:
        "실시간하한가"
        return float(self._rawbytes[46].decode("utf-8"))
    
    @property
    def dynm_prc_limt_yn(self) -> str:
        "실시간가격제한구분"
        return self._rawbytes[48].decode("utf-8")
    
@dataclass
class WSResp030(WSResp011):
    "주식선물 실시간호가[실시간-030]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0ZFASP0"

    @staticmethod
    def len() -> int:
        return 68
    
    @property
    def askp6(self) -> float:
        "매도호가6"
        return float(self._rawbytes[7].decode("utf-8"))
    
    @property
    def askp7(self) -> float:
        "매도호가7"
        return float(self._rawbytes[8].decode("utf-8"))
    
    @property
    def askp8(self) -> float:
        "매도호가8"
        return float(self._rawbytes[9].decode("utf-8"))
    
    @property
    def askp9(self) -> float:
        "매도호가9"
        return float(self._rawbytes[10].decode("utf-8"))
    
    @property
    def askp10(self) -> float:
        "매도호가10"
        return float(self._rawbytes[11].decode("utf-8"))
    
    @property
    def bidp1(self) -> float:
        "매수호가1"
        return float(self._rawbytes[12].decode("utf-8"))
    
    @property
    def bidp2(self) -> float:
        "매수호가2"
        return float(self._rawbytes[13].decode("utf-8"))
    
    @property
    def bidp3(self) -> float:
        "매수호가3"
        return float(self._rawbytes[14].decode("utf-8"))
    
    @property
    def bidp4(self) -> float:
        "매수호가4"
        return float(self._rawbytes[15].decode("utf-8"))
    
    @property
    def bidp5(self) -> float:
        "매수호가5"
        return float(self._rawbytes[16].decode("utf-8"))
    
    @property
    def bidp6(self) -> float:
        "매수호가6"
        return float(self._rawbytes[17].decode("utf-8"))
    
    @property
    def bidp7(self) -> float:
        "매수호가7"
        return float(self._rawbytes[18].decode("utf-8"))
    
    @property
    def bidp8(self) -> float:
        "매수호가8"
        return float(self._rawbytes[19].decode("utf-8"))
    
    @property
    def bidp9(self) -> float:
        "매수호가9"
        return float(self._rawbytes[20].decode("utf-8"))
    
    @property
    def bidp10(self) -> float:
        "매수호가10"
        return float(self._rawbytes[21].decode("utf-8"))
    
    @property
    def askp_csnu1(self) -> int:
        "매도호가건수1"
        return int(self._rawbytes[22].decode("utf-8"))
    
    @property
    def askp_csnu2(self) -> int:
        "매도호가건수2"
        return int(self._rawbytes[23].decode("utf-8"))
    
    @property
    def askp_csnu3(self) -> int:
        "매도호가건수3"
        return int(self._rawbytes[24].decode("utf-8"))
    
    @property
    def askp_csnu4(self) -> int:
        "매도호가건수4"
        return int(self._rawbytes[25].decode("utf-8"))
    
    @property
    def askp_csnu5(self) -> int:
        "매도호가건수5"
        return int(self._rawbytes[26].decode("utf-8"))
    
    @property
    def askp_csnu6(self) -> int:
        "매도호가건수6"
        return int(self._rawbytes[27].decode("utf-8"))
    
    @property
    def askp_csnu7(self) -> int:
        "매도호가건수7"
        return int(self._rawbytes[28].decode("utf-8"))
    
    @property
    def askp_csnu8(self) -> int:
        "매도호가건수8"
        return int(self._rawbytes[29].decode("utf-8"))
    
    @property
    def askp_csnu9(self) -> int:
        "매도호가건수9"
        return int(self._rawbytes[30].decode("utf-8"))
    
    @property
    def askp_csnu10(self) -> int:
        "매도호가건수10"
        return int(self._rawbytes[31].decode("utf-8"))
    
    @property
    def bidp_csnu1(self) -> int:
        "매수호가건수1"
        return int(self._rawbytes[32].decode("utf-8"))
    
    @property
    def bidp_csnu2(self) -> int:
        "매수호가건수2"
        return int(self._rawbytes[33].decode("utf-8"))
    
    @property
    def bidp_csnu3(self) -> int:
        "매수호가건수3"
        return int(self._rawbytes[34].decode("utf-8"))
    
    @property
    def bidp_csnu4(self) -> int:
        "매수호가건수4"
        return int(self._rawbytes[35].decode("utf-8"))
    
    @property
    def bidp_csnu5(self) -> int:
        "매수호가건수5"
        return int(self._rawbytes[36].decode("utf-8"))
    
    @property
    def bidp_csnu6(self) -> int:
        "매수호가건수6"
        return int(self._rawbytes[37].decode("utf-8"))
    
    @property
    def bidp_csnu7(self) -> int:
        "매수호가건수7"
        return int(self._rawbytes[38].decode("utf-8"))
    
    @property
    def bidp_csnu8(self) -> int:
        "매수호가건수8"
        return int(self._rawbytes[39].decode("utf-8"))
    
    @property
    def bidp_csnu9(self) -> int:
        "매수호가건수9"
        return int(self._rawbytes[40].decode("utf-8"))
    
    @property
    def bidp_csnu10(self) -> int:
        "매수호가건수10"
        return int(self._rawbytes[41].decode("utf-8"))
    
    @property
    def askp_rsqn1(self) -> int:
        "매도호가잔량1"
        return int(self._rawbytes[42].decode("utf-8"))
    
    @property
    def askp_rsqn2(self) -> int:
        "매도호가잔량2"
        return int(self._rawbytes[43].decode("utf-8"))
    
    @property
    def askp_rsqn3(self) -> int:
        "매도호가잔량3"
        return int(self._rawbytes[44].decode("utf-8"))
    
    @property
    def askp_rsqn4(self) -> int:
        "매도호가잔량4"
        return int(self._rawbytes[45].decode("utf-8"))
    
    @property
    def askp_rsqn5(self) -> int:
        "매도호가잔량5"
        return int(self._rawbytes[46].decode("utf-8"))
    
    @property
    def askp_rsqn6(self) -> int:
        "매도호가잔량6"
        return int(self._rawbytes[47].decode("utf-8"))
    
    @property
    def askp_rsqn7(self) -> int:
        "매도호가잔량7"
        return int(self._rawbytes[48].decode("utf-8"))
    
    @property
    def askp_rsqn8(self) -> int:
        "매도호가잔량8"
        return int(self._rawbytes[49].decode("utf-8"))
    
    @property
    def askp_rsqn9(self) -> int:
        "매도호가잔량9"
        return int(self._rawbytes[50].decode("utf-8"))
    
    @property
    def askp_rsqn10(self) -> int:
        "매도호가잔량10"
        return int(self._rawbytes[51].decode("utf-8"))
    
    @property
    def bidp_rsqn1(self) -> int:
        "매수호가잔량1"
        return int(self._rawbytes[52].decode("utf-8"))
    
    @property
    def bidp_rsqn2(self) -> int:
        "매수호가잔량2"
        return int(self._rawbytes[53].decode("utf-8"))
    
    @property
    def bidp_rsqn3(self) -> int:
        "매수호가잔량3"
        return int(self._rawbytes[54].decode("utf-8"))
    
    @property
    def bidp_rsqn4(self) -> int:
        "매수호가잔량4"
        return int(self._rawbytes[55].decode("utf-8"))
    
    @property
    def bidp_rsqn5(self) -> int:
        "매수호가잔량5"
        return int(self._rawbytes[56].decode("utf-8"))
    
    @property
    def bidp_rsqn6(self) -> int:
        "매수호가잔량6"
        return int(self._rawbytes[57].decode("utf-8"))
    
    @property
    def bidp_rsqn7(self) -> int:
        "매수호가잔량7"
        return int(self._rawbytes[58].decode("utf-8"))
    
    @property
    def bidp_rsqn8(self) -> int:
        "매수호가잔량8"
        return int(self._rawbytes[59].decode("utf-8"))
    
    @property
    def bidp_rsqn9(self) -> int:
        "매수호가잔량9"
        return int(self._rawbytes[60].decode("utf-8"))
    
    @property
    def bidp_rsqn10(self) -> int:
        "매수호가잔량10"
        return int(self._rawbytes[61].decode("utf-8"))
    
    @property
    def total_askp_csnu(self) -> int:
        "총매도호가건수"
        return int(self._rawbytes[62].decode("utf-8"))
    
    @property
    def total_bidp_csnu(self) -> int:
        "총매수호가건수"
        return int(self._rawbytes[63].decode("utf-8"))
    
    @property
    def total_askp_rsqn(self) -> int:
        "총매도호가잔량"
        return int(self._rawbytes[64].decode("utf-8"))
    
    @property
    def total_bidp_rsqn(self) -> int:
        "총매수호가잔량"
        return int(self._rawbytes[65].decode("utf-8"))
    
    @property
    def total_askp_rsqn_icdc(self) -> int:
        "총매도호가잔량증감"
        return int(self._rawbytes[66].decode("utf-8"))
    
    @property
    def total_bidp_rsqn_icdc(self) -> int:
        "총매수호가잔량증감"
        return int(self._rawbytes[67].decode("utf-8"))

@dataclass
class WSResp031(MetaWSResp):
    "주식선물 실시간예상체결 [실시간-031]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0ZFANC0"
    
    @staticmethod
    def len() -> int:
        return 8
    
    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")

    @property
    def bsop_hour(self) -> str:
        "영업시간"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def antc_cnpr(self) -> int:
        "예상체결가"
        return int(self._rawbytes[2].decode("utf-8"))
    
    @property
    def antc_cntg_vrss(self) -> int:
        "예상체결대비"
        return int(self._rawbytes[3].decode("utf-8"))
    
    @property
    def antc_cntg_vrss_sign(self) -> str:
        "예상체결대비부호"
        return self._rawbytes[4].decode("utf-8")
    
    @property
    def antc_cntg_prdy_ctrt(self) -> int:
        "예상체결전일대비율"
        return int(self._rawbytes[5].decode("utf-8"))
    
    @property
    def antc_mkop_cls_code(self) -> str:
        "예상장운영구분코드"
        return self._rawbytes[6].decode("utf-8")
    
    @property
    def antc_cnqn(self) -> int:
        "예상체결량"
        return int(self._rawbytes[7].decode("utf-8"))
    
@dataclass
class WSResp032(WSResp014):
    "EUREX야간옵션 실시간체결가 [실시간-032]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0EUCNT0"
    
    @staticmethod
    def len() -> int:
        return 53

    @property
    def avrg_vltl(self) -> float:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def dscs_lrqn_vol(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()

    @property
    def dynm_mxpr(self) -> float:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def dynm_llam(self) -> float:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def dynm_prc_limt_yn(self) -> str:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()

@dataclass
class WSResp033(WSResp015):
    "EUREX야간옵션 실시간호가 [실시간-033]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0EUASP0"

@dataclass
class WSResp034(WSResp031):
    "EUREX야간옵션실시간예상체결 [실시간-034]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0EUANC0"
    
    @staticmethod
    def len() -> int:
        return 7

    @property
    def antc_cnqn(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()

@dataclass
class WSResp041(WSResp003):
    "국내주식 실시간예상체결 [실시간-041]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0STANC0"
    
    @staticmethod
    def len() -> int:
        return 45

    @property
    def vi_stnd_prc(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()

@dataclass
class WSResp042(WSResp024):
    "국내주식 시간외 실시간체결가 [실시간-042]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0STOUP0"

@dataclass
class WSResp044(WSResp032):
    "주식옵션 실시간체결가 [실시간-044]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0ZOCNT0"

@dataclass
class WSResp045(WSResp011):
    "주식옵션 실시간호가 [실시간-045]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0ZOASP0"

    @staticmethod
    def len() -> int:
        return 68

    @property
    def askp6(self) -> float:
        "매도호가6"
        return float(self._rawbytes[38].decode("utf-8"))
    
    @property
    def askp7(self) -> float:
        "매도호가7"
        return float(self._rawbytes[39].decode("utf-8"))
    
    @property
    def askp8(self) -> float:
        "매도호가8"
        return float(self._rawbytes[40].decode("utf-8"))
    
    @property
    def askp9(self) -> float:
        "매도호가9"
        return float(self._rawbytes[41].decode("utf-8"))
    
    @property
    def askp10(self) -> float:
        "매도호가10"
        return float(self._rawbytes[42].decode("utf-8"))
    
    @property
    def bidp6(self) -> float:
        "매수호가6"
        return float(self._rawbytes[43].decode("utf-8"))
    
    @property
    def bidp7(self) -> float:
        "매수호가7"
        return float(self._rawbytes[44].decode("utf-8"))
    
    @property
    def bidp8(self) -> float:
        "매수호가8"
        return float(self._rawbytes[45].decode("utf-8"))
    
    @property
    def bidp9(self) -> float:
        "매수호가9"
        return float(self._rawbytes[46].decode("utf-8"))
    
    @property
    def bidp10(self) -> float:
        "매수호가10"
        return float(self._rawbytes[47].decode("utf-8"))
    
    @property
    def askp_csnu6(self) -> int:
        "매도호가건수6"
        return int(self._rawbytes[48].decode("utf-8"))
    
    @property
    def askp_csnu7(self) -> int:
        "매도호가건수7"
        return int(self._rawbytes[49].decode("utf-8"))
    
    @property
    def askp_csnu8(self) -> int:
        "매도호가건수8"
        return int(self._rawbytes[50].decode("utf-8"))
    
    @property
    def askp_csnu9(self) -> int:
        "매도호가건수9"
        return int(self._rawbytes[51].decode("utf-8"))
    
    @property
    def askp_csnu10(self) -> int:
        "매도호가건수10"
        return int(self._rawbytes[52].decode("utf-8"))
    
    @property
    def bidp_csnu6(self) -> int:
        "매수호가건수6"
        return int(self._rawbytes[53].decode("utf-8"))
    
    @property
    def bidp_csnu7(self) -> int:
        "매수호가건수7"
        return int(self._rawbytes[54].decode("utf-8"))
    
    @property
    def bidp_csnu8(self) -> int:
        "매수호가건수8"
        return int(self._rawbytes[55].decode("utf-8"))
    
    @property
    def bidp_csnu9(self) -> int:
        "매수호가건수9"
        return int(self._rawbytes[56].decode("utf-8"))
    
    @property
    def bidp_csnu10(self) -> int:
        "매수호가건수10"
        return int(self._rawbytes[57].decode("utf-8"))
    
    @property
    def askp_rsqn6(self) -> int:
        "매도호가잔량6"
        return int(self._rawbytes[58].decode("utf-8"))
    
    @property
    def askp_rsqn7(self) -> int:
        "매도호가잔량7"
        return int(self._rawbytes[59].decode("utf-8"))
    
    @property
    def askp_rsqn8(self) -> int:
        "매도호가잔량8"
        return int(self._rawbytes[60].decode("utf-8"))
    
    @property
    def askp_rsqn9(self) -> int:
        "매도호가잔량9"
        return int(self._rawbytes[61].decode("utf-8"))
    
    @property
    def askp_rsqn10(self) -> int:
        "매도호가잔량10"
        return int(self._rawbytes[62].decode("utf-8"))
    
    @property
    def bidp_rsqn6(self) -> int:
        "매수호가잔량6"
        return int(self._rawbytes[63].decode("utf-8"))
    
    @property
    def bidp_rsqn7(self) -> int:
        "매수호가잔량7"
        return int(self._rawbytes[64].decode("utf-8"))
    
    @property
    def bidp_rsqn8(self) -> int:
        "매수호가잔량8"
        return int(self._rawbytes[65].decode("utf-8"))
    
    @property
    def bidp_rsqn9(self) -> int:
        "매수호가잔량9"
        return int(self._rawbytes[66].decode("utf-8"))
    
    @property
    def bidp_rsqn10(self) -> int:
        "매수호가잔량10"
        return int(self._rawbytes[67].decode("utf-8"))
    
@dataclass
class WSResp046(WSResp034):
    "주식옵션 실시간예상체결 [실시간-046]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0ZOANC0"

@dataclass
class WSResp047(MetaWSResp):
    "국내주식 실시간회원사 [실시간-047]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0STMBC0"
    
    @staticmethod
    def len() -> int:
        return 78
    
    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def slen_mbcr_name1(self) -> str:
        "매도2회원사명1"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def slen_mbcr_name2(self) -> str:
        "매도회원사명2"
        return self._rawbytes[2].decode("utf-8")
    
    @property
    def slen_mbcr_name3(self) -> str:
        "매도회원사명3"
        return self._rawbytes[3].decode("utf-8")
    
    @property
    def slen_mbcr_name4(self) -> str:
        "매도회원사명4"
        return self._rawbytes[4].decode("utf-8")
    
    @property
    def slen_mbcr_name5(self) -> str:
        "매도회원사명5"
        return self._rawbytes[5].decode("utf-8")
    
    @property
    def shnu_mbcr_name1(self) -> str:
        "매수회원사명1"
        return self._rawbytes[6].decode("utf-8")
    
    @property
    def shnu_mbcr_name2(self) -> str:
        "매수회원사명2"
        return self._rawbytes[7].decode("utf-8")
    
    @property
    def shnu_mbcr_name3(self) -> str:
        "매수회원사명3"
        return self._rawbytes[8].decode("utf-8")
    
    @property
    def shnu_mbcr_name4(self) -> str:
        "매수회원사명4"
        return self._rawbytes[9].decode("utf-8")
    
    @property
    def shnu_mbcr_name5(self) -> str:
        "매수회원사명5"
        return self._rawbytes[10].decode("utf-8")
    
    @property
    def total_seln_qty1(self) -> int:
        "총매도수량1"
        return int(self._rawbytes[11].decode("utf-8"))
    
    @property
    def total_seln_qty2(self) -> int:
        "총매도수량2"
        return int(self._rawbytes[12].decode("utf-8"))
    
    @property
    def total_seln_qty3(self) -> int:
        "총매도수량3"
        return int(self._rawbytes[13].decode("utf-8"))
    
    @property
    def total_seln_qty4(self) -> int:
        "총매도수량4"
        return int(self._rawbytes[14].decode("utf-8"))
    
    @property
    def total_seln_qty5(self) -> int:
        "총매도수량5"
        return int(self._rawbytes[15].decode("utf-8"))
    
    @property
    def total_shnu_qty1(self) -> int:
        "총매수수량1"
        return int(self._rawbytes[16].decode("utf-8"))
    
    @property
    def total_shnu_qty2(self) -> int:
        "총매수수량2"
        return int(self._rawbytes[17].decode("utf-8"))
    
    @property
    def total_shnu_qty3(self) -> int:
        "총매수수량3"
        return int(self._rawbytes[18].decode("utf-8"))
    
    @property
    def total_shnu_qty4(self) -> int:
        "총매수수량4"
        return int(self._rawbytes[19].decode("utf-8"))
    
    @property
    def total_shnu_qty5(self) -> int:
        "총매수수량5"
        return int(self._rawbytes[20].decode("utf-8"))
    
    @property
    def seln_mbcr_glob_yn1(self) -> str:
        "매도거래원구분1"
        return self._rawbytes[21].decode("utf-8")

    @property
    def seln_mbcr_glob_yn2(self) -> str:
        "매도거래원구분2"
        return self._rawbytes[22].decode("utf-8")
    
    @property
    def seln_mbcr_glob_yn3(self) -> str:
        "매도거래원구분3"
        return self._rawbytes[23].decode("utf-8")
    
    @property
    def seln_mbcr_glob_yn4(self) -> str:
        "매도거래원구분4"
        return self._rawbytes[24].decode("utf-8")
    
    @property
    def seln_mbcr_glob_yn5(self) -> str:
        "매도거래원구분5"
        return self._rawbytes[25].decode("utf-8")
    
    @property
    def shnu_mbcr_glob_yn1(self) -> str:
        "매수거래원구분1"
        return self._rawbytes[26].decode("utf-8")
    
    @property
    def shnu_mbcr_glob_yn2(self) -> str:
        "매수거래원구분2"
        return self._rawbytes[27].decode("utf-8")
    
    @property
    def shnu_mbcr_glob_yn3(self) -> str:
        "매수거래원구분3"
        return self._rawbytes[28].decode("utf-8")
    
    @property
    def shnu_mbcr_glob_yn4(self) -> str:
        "매수거래원구분4"
        return self._rawbytes[29].decode("utf-8")
    
    @property
    def shnu_mbcr_glob_yn5(self) -> str:
        "매수거래원구분5"
        return self._rawbytes[30].decode("utf-8")
    
    @property
    def slen_mbcr_no1(self) -> str:
        "매도거래원코드1"
        return self._rawbytes[31].decode("utf-8")
    
    @property
    def slen_mbcr_no2(self) -> str:
        "매도거래원코드2"
        return self._rawbytes[32].decode("utf-8")
    
    @property
    def slen_mbcr_no3(self) -> str:
        "매도거래원코드3"
        return self._rawbytes[33].decode("utf-8")
    
    @property
    def slen_mbcr_no4(self) -> str:
        "매도거래원코드4"
        return self._rawbytes[34].decode("utf-8")
    
    @property
    def slen_mbcr_no5(self) -> str:
        "매도거래원코드5"
        return self._rawbytes[35].decode("utf-8")
    
    @property
    def shnu_mbcr_no1(self) -> str:
        "매수거래원코드1"
        return self._rawbytes[36].decode("utf-8")
    
    @property
    def shnu_mbcr_no2(self) -> str:
        "매수거래원코드2"
        return self._rawbytes[37].decode("utf-8")
    
    @property
    def shnu_mbcr_no3(self) -> str:
        "매수거래원코드3"
        return self._rawbytes[38].decode("utf-8")
    
    @property
    def shnu_mbcr_no4(self) -> str:
        "매수거래원코드4"
        return self._rawbytes[39].decode("utf-8")
    
    @property
    def shnu_mbcr_no5(self) -> str:
        "매수거래원코드5"
        return self._rawbytes[40].decode("utf-8")
    
    @property
    def seln_mbcr_rlim1(self) -> int:
        "매도회원사비중1"
        return int(self._rawbytes[41].decode("utf-8"))

    @property
    def seln_mbcr_rlim2(self) -> int:
        "매도회원사비중2"
        return int(self._rawbytes[42].decode("utf-8"))
    
    @property
    def seln_mbcr_rlim3(self) -> int:
        "매도회원사비중3"
        return int(self._rawbytes[43].decode("utf-8"))
    
    @property
    def seln_mbcr_rlim4(self) -> int:
        "매도회원사비중4"
        return int(self._rawbytes[44].decode("utf-8"))
    
    @property
    def seln_mbcr_rlim5(self) -> int:
        "매도회원사비중5"
        return int(self._rawbytes[45].decode("utf-8"))
    
    @property
    def shnu_mbcr_rlim1(self) -> int:
        "매수회원사비중1"
        return int(self._rawbytes[46].decode("utf-8"))
    
    @property
    def shnu_mbcr_rlim2(self) -> int:
        "매수회원사비중2"
        return int(self._rawbytes[47].decode("utf-8"))
    
    @property
    def shnu_mbcr_rlim3(self) -> int:
        "매수회원사비중3"
        return int(self._rawbytes[48].decode("utf-8"))
    
    @property
    def shnu_mbcr_rlim4(self) -> int:
        "매수회원사비중4"
        return int(self._rawbytes[49].decode("utf-8"))
    
    @property
    def shnu_mbcr_rlim5(self) -> int:
        "매수회원사비중5"
        return int(self._rawbytes[50].decode("utf-8"))
    
    @property
    def seln_qty_icdc1(self) -> int:
        "매도수량증감1"
        return int(self._rawbytes[51].decode("utf-8"))
    
    @property
    def seln_qty_icdc2(self) -> int:
        "매도수량증감2"
        return int(self._rawbytes[52].decode("utf-8"))
    
    @property
    def seln_qty_icdc3(self) -> int:
        "매도수량증감3"
        return int(self._rawbytes[53].decode("utf-8"))
    
    @property
    def seln_qty_icdc4(self) -> int:
        "매도수량증감4"
        return int(self._rawbytes[54].decode("utf-8"))
    
    @property
    def seln_qty_icdc5(self) -> int:
        "매도수량증감5"
        return int(self._rawbytes[55].decode("utf-8"))
    
    @property
    def shnu_qty_icdc1(self) -> int:
        "매수수량증감1"
        return int(self._rawbytes[56].decode("utf-8"))
    
    @property
    def shnu_qty_icdc2(self) -> int:
        "매수수량증감2"
        return int(self._rawbytes[57].decode("utf-8"))
    
    @property
    def shnu_qty_icdc3(self) -> int:
        "매수수량증감3"
        return int(self._rawbytes[58].decode("utf-8"))
    
    @property
    def shnu_qty_icdc4(self) -> int:
        "매수수량증감4"
        return int(self._rawbytes[59].decode("utf-8"))
    
    @property
    def shnu_qty_icdc5(self) -> int:
        "매수수량증감5"
        return int(self._rawbytes[60].decode("utf-8"))
    
    @property
    def glob_total_seln_qty(self) -> int:
        "외국계총매도수량"
        return int(self._rawbytes[61].decode("utf-8"))

    @property
    def glob_total_shnu_qty(self) -> int:
        "외국계총매수수량"
        return int(self._rawbytes[62].decode("utf-8"))
    
    @property
    def glob_total_seln_qty_icdc(self) -> int:
        "외국계총매도수량증감"
        return int(self._rawbytes[63].decode("utf-8"))
    
    @property
    def glob_total_shnu_qty_icdc(self) -> int:
        "외국계총매수수량증감"
        return int(self._rawbytes[64].decode("utf-8"))
    
    @property
    def glob_ntby_qty(self) -> int:
        "외국계순매수수량"
        return int(self._rawbytes[65].decode("utf-8"))
    
    @property
    def glob_seln_rlim(self) -> int:
        "외국계매도비중"
        return int(self._rawbytes[66].decode("utf-8"))

    @property
    def glob_shnu_rlim(self) -> int:
        "외국계매수비중"
        return int(self._rawbytes[67].decode("utf-8"))
    
    @property
    def seln_mbcr_eng_name1(self) -> str:
        "매도회원사영문명1"
        return self._rawbytes[68].decode("utf-8")
    
    @property
    def seln_mbcr_eng_name2(self) -> str:
        "매도회원사영문명2"
        return self._rawbytes[69].decode("utf-8")
    
    @property
    def seln_mbcr_eng_name3(self) -> str:
        "매도회원사영문명3"
        return self._rawbytes[70].decode("utf-8")
    
    @property
    def seln_mbcr_eng_name4(self) -> str:
        "매도회원사영문명4"
        return self._rawbytes[71].decode("utf-8")
    
    @property
    def seln_mbcr_eng_name5(self) -> str:
        "매도회원사영문명5"
        return self._rawbytes[72].decode("utf-8")
    
    @property
    def shnu_mbcr_eng_name1(self) -> str:
        "매수회원사영문명1"
        return self._rawbytes[73].decode("utf-8")
    
    @property
    def shnu_mbcr_eng_name2(self) -> str:
        "매수회원사영문명2"
        return self._rawbytes[74].decode("utf-8")
    
    @property
    def shnu_mbcr_eng_name3(self) -> str:
        "매수회원사영문명3"
        return self._rawbytes[75].decode("utf-8")
    
    @property
    def shnu_mbcr_eng_name4(self) -> str:
        "매수회원사영문명4"
        return self._rawbytes[76].decode("utf-8")
    
    @property
    def shnu_mbcr_eng_name5(self) -> str:
        "매수회원사영문명5"
        return self._rawbytes[77].decode("utf-8")

@dataclass
class WSResp048(MetaWSResp):
    "국내주식 실시간프로그램매매 (KRX)[실시간-048]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0STPGM0"
    
    @staticmethod
    def len() -> int:
        return 11

    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def cntg_hour(self) -> str:
        "체결시간"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def seln_cnqn(self) -> int:
        "매도체결량"
        return int(self._rawbytes[2].decode("utf-8"))
    
    @property
    def seln_tr_pbmn(self) -> int:
        "매도체결대금"
        return int(self._rawbytes[3].decode("utf-8"))
    
    @property
    def shnu_cnqn(self) -> int:
        "매수체결량"
        return int(self._rawbytes[4].decode("utf-8"))
    
    @property
    def shnu_tr_pbmn(self) -> int:
        "매수체결대금"
        return int(self._rawbytes[5].decode("utf-8"))
    
    @property
    def ntby_cnqn(self) -> int:
        "순매수체결량"
        return int(self._rawbytes[6].decode("utf-8"))
    
    @property
    def ntby_tr_pbmn(self) -> int:
        "순매수체결대금"
        return int(self._rawbytes[7].decode("utf-8"))
    
    @property
    def seln_rsqn(self) -> int:
        "매도호가잔량"
        return int(self._rawbytes[8].decode("utf-8"))
    
    @property
    def shnu_rsqn(self) -> int:
        "매수호가잔량"
        return int(self._rawbytes[9].decode("utf-8"))
    
    @property
    def whol_ntby_rsqn(self) -> int:
        "전체순매수호가잔량"
        return int(self._rawbytes[10].decode("utf-8"))
    
@dataclass
class WSResp049(MetaWSResp):
    "국내주식 장운영정보 [실시간-049]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0STMKO0"
    
    @staticmethod
    def len() -> int:
        return 10

    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def trht_yn(self) -> str:
        "거래정지여부"
        return self._rawbytes[1].decode("utf-8")

    @property
    def tr_susp_reas_cntt(self) -> str:
        "거래정지사유내용"
        return self._rawbytes[2].decode("utf-8")
    
    @property
    def mkop_cls_code(self) -> str:
        "장운영구분코드"
        return self._rawbytes[3].decode("utf-8")
    
    @property
    def antc_mkop_cls_code(self) -> str:
        "예상장운영구분코드"
        return self._rawbytes[4].decode("utf-8")
    
    @property
    def mrkt_trtm_cls_code(self) -> str:
        "임의연장구분코드"
        return self._rawbytes[5].decode("utf-8")

    @property
    def divi_app_cls_code(self) -> str:
        "동시호가배분처리구분코드"
        return self._rawbytes[6].decode("utf-8")

    @property
    def iscd_stat_cls_code(self) -> str:
        "종목상태구분코드"
        return self._rawbytes[7].decode("utf-8")

    @property
    def vi_cls_code(self) -> str:
        "VI적용구분코드"
        return self._rawbytes[8].decode("utf-8")
    
    @property
    def ovtm_vi_cls_code(self) -> str:
        "시간외VI적용구분코드"
        return self._rawbytes[9].decode("utf-8")
    
@dataclass
class WSResp051(MetaWSResp):
    "국내ETF NAV추이 [실시간-051]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0STNAV0"

    @staticmethod
    def len() -> int:
        return 8
    
    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def nav(self) -> float:
        "NAV"
        return float(self._rawbytes[1].decode("utf-8"))
    
    @property
    def prdy_vrss_sign(self) -> str:
        "전일대비부호"
        return self._rawbytes[2].decode("utf-8")
    
    @property
    def prdy_vrss(self) -> float:
        "전일대비"
        return float(self._rawbytes[3].decode("utf-8"))
    
    @property
    def prdy_ctrt(self) -> float:
        "전일대비율"
        return float(self._rawbytes[4].decode("utf-8"))
    
    @property
    def oprc_nav(self) -> float:
        "시가NAV"
        return float(self._rawbytes[5].decode("utf-8"))
    
    @property
    def hprc_nav(self) -> float:
        "고가NAV"
        return float(self._rawbytes[6].decode("utf-8"))
    
    @property
    def lprc_nav(self) -> float:
        "저가NAV"
        return float(self._rawbytes[7].decode("utf-8"))
    
@dataclass
class WSResp061(MetaWSResp):
    "ELW 실시간체결가 [실시간-061]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0EWCNT0"
    
    @staticmethod
    def len() -> int:
        return 63
    
    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[0].decode("utf-8")
    
    @property
    def cntg_hour(self) -> str:
        "체결시간"
        return self._rawbytes[1].decode("utf-8")
    
    @property
    def prpr(self) -> float:
        "현재가"
        return float(self._rawbytes[2].decode("utf-8"))

    @property
    def prdy_vrss_sign(self) -> str:
        "전일대비부호"
        return self._rawbytes[3].decode("utf-8")
    
    @property
    def prdy_vrss(self) -> float:
        "전일대비"
        return float(self._rawbytes[4].decode("utf-8"))
    
    @property
    def prdy_ctrt(self) -> float:
        "전일대비율"
        return float(self._rawbytes[5].decode("utf-8"))
    
    @property
    def wghn_avrg_prc(self) -> float:
        "가중평균가"
        return float(self._rawbytes[6].decode("utf-8"))
    
    @property
    def oprc(self) -> float:
        "시가"
        return float(self._rawbytes[7].decode("utf-8"))
    
    @property
    def hprc(self) -> float:
        "고가"
        return float(self._rawbytes[8].decode("utf-8"))
    
    @property
    def lprc(self) -> float:
        "저가"
        return float(self._rawbytes[9].decode("utf-8"))
    
    @property
    def askp1(self) -> float:
        "매도호가1"
        return float(self._rawbytes[10].decode("utf-8"))
    
    @property
    def bidp1(self) -> float:
        "매수호가1"
        return float(self._rawbytes[11].decode("utf-8"))
    
    @property
    def cntg_vol(self) -> int:
        "체결수량"
        return int(self._rawbytes[12].decode("utf-8"))

    @property
    def acml_vol(self) -> int:
        "누적거래량"
        return int(self._rawbytes[13].decode("utf-8"))
    
    @property
    def acml_tr_pbmn(self) -> int:
        "누적거래대금"
        return int(self._rawbytes[14].decode("utf-8"))
    
    @property
    def seln_cntg_csnu(self) -> int:
        "매도체결건수"
        return int(self._rawbytes[15].decode("utf-8"))
    
    @property
    def shnu_cntg_csnu(self) -> int:
        "매수체결건수"
        return int(self._rawbytes[16].decode("utf-8"))
    
    @property
    def ntby_cntg_csnu(self) -> int:
        "순매수체결건수"
        return int(self._rawbytes[17].decode("utf-8"))
    
    @property
    def cttr(self) -> float:
        "체결강도"
        return float(self._rawbytes[18].decode("utf-8"))
    
    @property
    def slen_cntg_smtn(self) -> int:
        "총매도수량"
        return int(self._rawbytes[19].decode("utf-8"))
    
    @property
    def shnu_cntg_smtn(self) -> int:
        "총매수수량"
        return int(self._rawbytes[20].decode("utf-8"))
    
    @property
    def cntg_cls_code(self) -> str:
        "체결구분코드"
        return self._rawbytes[21].decode("utf-8")
    
    @property
    def shnu_rate(self) -> float:
        "매수비율"
        return float(self._rawbytes[22].decode("utf-8"))
    
    @property
    def prdy_vol_vrss_acml_vol_rate(self) -> float:
        "전일거래량대비등락율"
        return float(self._rawbytes[23].decode("utf-8"))
    
    @property
    def oprc_hour(self) -> str:
        "시가시간"
        return self._rawbytes[24].decode("utf-8")
    
    @property
    def oprc_vrss_prpr_sign(self) -> str:
        "시가대비부호"
        return self._rawbytes[25].decode("utf-8")
    
    @property
    def oprc_vrss_prpr(self) -> float:
        "시가대비"
        return float(self._rawbytes[26].decode("utf-8"))
    
    @property
    def hgpr_hour(self) -> str:
        "고가시간"
        return self._rawbytes[27].decode("utf-8")
    
    @property
    def hgpr_vrss_prpr_sign(self) -> str:
        "고가대비부호"
        return self._rawbytes[28].decode("utf-8")
    
    @property
    def hgpr_vrss_prpr(self) -> float:
        "고가대비"
        return float(self._rawbytes[29].decode("utf-8"))
    
    @property
    def lgpr_hour(self) -> str:
        "저가시간"
        return self._rawbytes[30].decode("utf-8")
    
    @property
    def lgpr_vrss_prpr_sign(self) -> str:
        "저가대비부호"
        return self._rawbytes[31].decode("utf-8")
    
    @property
    def lgpr_vrss_prpr(self) -> float:
        "저가대비"
        return float(self._rawbytes[32].decode("utf-8"))
    
    @property
    def bsop_date(self) -> str:
        "영업일자"
        return self._rawbytes[33].decode("utf-8")

    @property
    def new_mkop_cls_code(self) -> str:
        "(신)장운영구분코드"
        return self._rawbytes[34].decode("utf-8")

    @property
    def trht_yn(self) -> str:
        "거래정지여부"
        return self._rawbytes[35].decode("utf-8")

    @property
    def askp1_rsqn1(self) -> int:
        "매도호가잔량1"
        return int(self._rawbytes[36].decode("utf-8"))
    
    @property
    def bidp1_rsqn1(self) -> int:
        "매수호가잔량1"
        return int(self._rawbytes[37].decode("utf-8"))
    
    @property
    def total_askp_rsqn(self) -> int:
        "총매도호가잔량"
        return int(self._rawbytes[38].decode("utf-8"))
    
    @property
    def total_bidp_rsqn(self) -> int:
        "총매수호가잔량"
        return int(self._rawbytes[39].decode("utf-8"))
    
    @property
    def tmvl_val(self) -> float:
        "시간가치값"
        return float(self._rawbytes[40].decode("utf-8"))

    @property
    def prit(self) -> float: 
        "패리티"
        return float(self._rawbytes[41].decode("utf-8"))
    
    @property
    def prmm_val(self) -> float:
        "프리미엄값"
        return float(self._rawbytes[42].decode("utf-8"))
    
    @property
    def gear(self) -> float:
        "기어링"
        return float(self._rawbytes[43].decode("utf-8"))
    
    @property
    def prls_qryr_rate(self) -> float:
        "손익분기비율"
        return float(self._rawbytes[44].decode("utf-8"))
    
    @property
    def invl_val(self) -> float:
        "내재가치값"
        return float(self._rawbytes[45].decode("utf-8"))
    
    @property
    def prmm_rate(self) -> float:
        "프리미엄비율"
        return float(self._rawbytes[46].decode("utf-8"))

    @property
    def cfp(self) -> float:
        "자본지지점"
        return float(self._rawbytes[47].decode("utf-8"))

    @property
    def lvrg_val(self) -> float:
        "레버리지값"
        return float(self._rawbytes[48].decode("utf-8"))
    
    @property
    def delta(self) -> float:
        "Greeks-델타"
        return float(self._rawbytes[49].decode("utf-8"))
    
    @property
    def gamma(self) -> float:
        "Greeks-감마"
        return float(self._rawbytes[50].decode("utf-8"))
    
    @property
    def vega(self) -> float:
        "Greeks-베가"
        return float(self._rawbytes[51].decode("utf-8"))

    @property
    def theta(self) -> float:
        "Greeks-세타"
        return float(self._rawbytes[52].decode("utf-8"))
    
    @property
    def rho(self) -> float:
        "Greeks-로"
        return float(self._rawbytes[53].decode("utf-8"))
    
    @property
    def hts_ints_vltl(self) -> float:
        "HTS내재변동성"
        return float(self._rawbytes[54].decode("utf-8"))

    @property
    def hts_thpr(self) -> float:
        "HTS시간이론가"
        return float(self._rawbytes[55].decode("utf-8"))

    @property
    def vol_tnrt(self) -> float:
        "거래량회전율"
        return float(self._rawbytes[56].decode("utf-8"))
    
    @property
    def prdy_smns_hour_acml_vol(self) -> int:
        "전일동시간누적거래량"
        return int(self._rawbytes[57].decode("utf-8"))
    
    @property
    def prdy_smns_hour_acml_vol_rate(self) -> float:
        "전일동시간누적거래량대비율"
        return float(self._rawbytes[58].decode("utf-8"))
    
    @property
    def apprch_rate(self) -> float:
        "접근도"
        return float(self._rawbytes[59].decode("utf-8"))
    
    @property
    def lp_hvol(self) -> int:
        "LP보유량"
        return int(self._rawbytes[60].decode("utf-8"))

    @property
    def lp_hldn_rate(self) -> float:
        "LP보유비율"
        return float(self._rawbytes[61].decode("utf-8"))
    
    @property
    def lp_ntby_qty(self) -> int:
        "LP순매수량? (문서에는 순매도로 나옴 확인필요)"
        return int(self._rawbytes[62].decode("utf-8"))

@dataclass
class WSResp062(WSResp004):
    """
    ELW 실시간호가 [실시간-062]
    
    - Note: LP 호가잔량 부분 API문서 불확실로 인해 구현하지 않음
    """
    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0EWASP0"
    
    @staticmethod
    def len() -> int:
        return 73
    
    @property
    def antc_cnpr(self) -> int:
        "예상체결가"
        return int(self._rawbytes[45].decode("utf-8"))
    
    @property
    def antc_cnqn(self) -> int:
        "예상체결량"
        return int(self._rawbytes[46].decode("utf-8"))
    
    @property
    def antc_cntg_vrss_sign(self) -> str:
        "예상체결대비부호"
        return self._rawbytes[47].decode("utf-8")
    
    @property
    def antc_cntg_vrss(self) -> int:
        "예상체결대비"
        return int(self._rawbytes[48].decode("utf-8"))
    
    @property
    def antc_cntg_prdy_ctrt(self) -> float:
        "예상체결전일대비율"
        return float(self._rawbytes[49].decode("utf-8"))
    
    @property
    def antc_vol(self) -> int:
        "예상거래량"
        return int(self._rawbytes[72].decode("utf-8"))
    
    @property
    def ovtm_total_askp_rsqn(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def ovtm_total_bidp_rsqn(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()

    @property
    def acml_vol(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def total_askp_rsqn_icdc(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def total_bidp_rsqn_icdc(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def ovtm_total_askp_icdc(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def ovtm_total_bidp_icdc(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
@dataclass
class WSResp063(WSResp061):
    "ELW 실시간예상체결 [실시간-063]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0EWASP0"
    
    @staticmethod
    def len() -> int:
        return 59
    
    @property
    def lp_hvol(self) -> int:
        "LP보유량"
        return int(self._rawbytes[57].decode("utf-8"))
    
    @property
    def lp_hldn_rate(self) -> float:
        "LP보유비율"
        return float(self._rawbytes[58].decode("utf-8"))

    @property
    def prdy_smns_hour_acml_vol(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def prdy_smns_hour_acml_vol_rate(self) -> float:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def apprch_rate(self) -> float:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def lp_ntby_qty(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()

@dataclass
class WSResp064(WSResp010):
    "CME야간선물 실시간종목체결 [실시간-064]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0MFCNT0"
    
    @staticmethod
    def len() -> int:
        return 46
    
    @property
    def dscs_bltr_acml_qty(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
        return int(self._rawbytes[46].decode("utf-8"))
    
    @property
    def dynm_mxpr(self) -> float:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def dynm_llam(self) -> float:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def dynm_prc_limt_yn(self) -> str:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()

@dataclass
class WSResp065(WSResp011):
    "CME야간선물 실시간호가 [실시간-065]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0MFASP0"

@dataclass
class WSResp066(WSResp005):
    "CME야간선물 실시간체결통보 [실시간-066]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0MFCNI0"
    
    @staticmethod
    def len() -> int:
        return 19

    @property
    def shrn_iscd(self) -> str:
        "단축종목코드"
        return self._rawbytes[7].decode("utf-8")
    
    @property
    def cntg_qty(self) -> int:
        "체결수량"
        return int(self._rawbytes[8].decode("utf-8"))
    
    @property
    def cntg_unpr(self) -> int:
        "체결단가"
        return int(self._rawbytes[9].decode("utf-8"))
    
    @property
    def cntg_hour(self) -> str:
        "체결시간"
        return self._rawbytes[10].decode("utf-8")
    
    @property
    def rfus_yn(self) -> str:
        "거부여부"
        return self._rawbytes[11].decode("utf-8")
    
    @property
    def cntg_yn(self) -> str:
        "체결여부"
        return self._rawbytes[12].decode("utf-8")
    
    @property
    def acpt_yn(self) -> str:
        "접수여부"
        return self._rawbytes[13].decode("utf-8")
    
    @property
    def brnc_no(self) -> str:
        "지점번호"
        return self._rawbytes[14].decode("utf-8")
    
    @property
    def oder_qty(self) -> int:
        "주문수량"
        return int(self._rawbytes[15].decode("utf-8"))
    
    @property
    def acnt_name(self) -> str:
        "계좌명"
        return self._rawbytes[16].decode("utf-8")
    
    @property
    def cntg_isnm(self) -> str:
        "체결종목명"
        return self._rawbytes[17].decode("utf-8")
    
    @property
    def oder_cond(self) -> str:
        "주문조건"
        return self._rawbytes[18].decode("utf-8")
    
    @property
    def crdt_cls(self) -> str:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def crdt_loan_date(self) -> str:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def cntg_isnm40(self) -> str:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
    @property
    def oder_prc(self) -> int:
        "사용안함 - NotImplementedError"
        raise NotImplementedError()
    
@dataclass
class WSResp067(WSResp066):
    "EUREX야간옵션실시간체결통보 [실시간-067]"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0EUCNI0"

@dataclass
class WSResp003NXT(WSResp003):
    "국내주식 실시간체결가 (NXT)"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0UNCNT0"

@dataclass
class WSResp004NXT(WSResp004):
    "국내주식 실시간호가 (NXT)"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0UNASP0"

@dataclass
class WSResp048NXT(WSResp048):
    "국내주식 실시간프로그램매매 (NXT)"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0UNPGM0"

@dataclass
class WSResp003UNI(WSResp003):
    "국내주식 실시간체결가 (UNI)"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0UNCNT0"
    
@dataclass
class WSResp004UNI(WSResp004):
    "국내주식 실시간호가 (UNI)"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0UNASP0"
    
    @staticmethod
    def len() -> int:
        return 66
    
    @property
    def stck_deal_cls_code(self) -> str:
        "주식거래구분코드"
        return self._rawbytes[59].decode("utf-8")
    
    @property
    def kmid_prc(self) -> float:
        "KRX 중간가"
        return float(self._rawbytes[60].decode("utf-8"))
    
    @property
    def kmid_total_rsqn(self) -> int:
        "KRX 중간가총잔량"
        return int(self._rawbytes[61].decode("utf-8"))
    
    @property
    def kmid_cls_code(self) -> str:
        "KRX 중간가 매수매도 구분"
        return self._rawbytes[62].decode("utf-8")
    
    @property
    def nmid_prc(self) -> float:
        "NXT 중간가"
        return float(self._rawbytes[63].decode("utf-8"))
    
    @property
    def nmid_total_rsqn(self) -> int:
        "NXT 중간가총잔량"
        return int(self._rawbytes[64].decode("utf-8"))
    
    @property
    def nmid_cls_code(self) -> str:
        "NXT 중간가 매수매도 구분"
        return self._rawbytes[65].decode("utf-8")
    
@dataclass
class WSResp048UNI(WSResp048):
    "국내주식 실시간프로그램매매 (UNI)"

    @staticmethod
    def trid(ismock: bool=False) -> str:
        return "H0UNPGM0"

