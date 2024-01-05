
import win32com.client


### CpUtil.CpCodeMgr ###
class ObjCpCodeMgr:

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=287&seq=11&page=1&searchString=%ec%9e%a5&p=&v=&m=
    """

    obj = win32com.client.Dispatch("CpUtil.CpCodeMgr")
    
    @classmethod
    def code_to_name(cls, code):
        return cls.obj.CodeToName(code)
    
    @classmethod
    def get_stock_margin_rate(cls, code):
        return cls.obj.GetStockMarginRate(code)
    
    @classmethod
    def get_stock_trade_min(cls, code):
        return cls.obj.GetStockMemeMin(code)
    
    @classmethod
    def get_stock_industry_code(cls, code):
        return cls.obj.GetStockIndustryCode (code)
    
    @classmethod
    def get_stock_market_kind(cls, code):
        return cls.obj.GetStockMarketKind(code)
    
    @classmethod
    def get_stock_control_kind(cls, code):
        return cls.obj.GetStockControlKind(code)

    @classmethod
    def get_stock_over_heating(cls, code):
        return cls.obj.GetOverHeating(code)
    
    @classmethod
    def get_stock_trade_delist(cls, code):
        return cls.obj.IsStockArrgSby(code)
    
    @classmethod
    def is_etp_warning(cls, code):
        return cls.obj.IsStockIoi(code)
    
    @classmethod
    def get_stock_supervision_kind(cls, code):
        return cls.obj.GetStockSupervisionKind (code)
    
    @classmethod
    def get_stock_status_kind(cls, code):
        return cls.obj.GetStockStatusKind (code)
    
    @classmethod
    def get_stock_capital_size(cls, code):
        return cls.obj.GetStockCapital(code)
    
    @classmethod
    def get_stock_fiscal_month(cls, code):
        return cls.obj.GetStockFiscalMonth(code)
    
    @classmethod
    def get_stock_conglomerate_code(cls, code):
        return cls.obj.GetStockGroupCode(code)
    
    @classmethod
    def get_stock_is_kospi200(cls, code):
        return cls.obj.GetStockKospi200Kind(code)
    
    @classmethod
    def get_stock_type(cls, code):
        return cls.obj.GetStockSectionKind(code)
    
    @classmethod
    def get_stock_lac_kind(cls, code):
        return cls.obj.GetStockLacKind(code)
    
    @classmethod
    def get_stock_listed_date(cls, code):
        return cls.obj.GetStockListedDate(code)

    @classmethod
    def get_stock_max_price(cls, code):
        return cls.obj.GetStockMaxPrice(code)
    
    @classmethod
    def get_stock_min_price(cls, code):
        return cls.obj.GetStockMinPrice(code)
    
    @classmethod
    def get_stock_par_price(cls, code):
        return cls.obj.GetStockParPrice(code)
    
    @classmethod
    def get_stock_std_price(cls, code):
        return cls.obj.GetStockStdPrice(code)
    
    @classmethod
    def get_stock_yesterday_open(cls, code):
        return cls.obj.GetStockYdOpenPrice(code)
    
    @classmethod
    def get_stock_yesterday_high(cls, code):
        return cls.obj.GetStockYdHighPrice(code)
    
    @classmethod
    def get_stock_yesterday_low(cls, code):
        return cls.obj.GetStockYdLowPrice(code)
    
    @classmethod
    def get_stock_yesterday_close(cls, code):
        return cls.obj.GetStockYdClosePrice(code)
    
    @classmethod
    def is_creditable(cls, code):
        return cls.obj.IsStockCreditEnable(code)

    @classmethod
    def get_stock_parprice_change(cls, code):
        return cls.obj.GetStockParPriceChageType(code)

    @classmethod
    def is_spac(cls, code):
        return cls.obj.IsSPAC(code)

    @classmethod
    def get_mini_future_list(cls):
        return cls.obj.GetMiniFutureList()

    @classmethod
    def get_mini_option_list(cls):
        return cls.obj.GetMiniOptionList()

    @classmethod
    def reload_port_data(cls):
        return cls.obj.ReLoadPortData()

    @classmethod
    def is_big_listing(cls, code):
        return cls.obj.IsBigListingStock(code)
    
    @classmethod
    def get_stock_list_by_market(cls, market_code):
        return cls.obj.GetStockListByMarket(market_code)
    
    @classmethod
    def get_group_code_list(cls, group_code):
        return cls.obj.GetGroupCodeList(group_code)
    
    @classmethod
    def get_group_name(cls, group_code):
        return cls.obj.GetGroupName(group_code)
    
    @classmethod
    def get_industry_list(cls):
        return cls.obj.GetIndustryList()
    
    @classmethod
    def get_industry_name(cls, industry_code):
        return cls.obj.GetIndustryName(industry_code)
    
    @classmethod
    def get_member_list(cls):
        return cls.obj.GetMemberList()
    
    @classmethod
    def get_member_name(cls, member_code):
        return cls.obj.GetMemberName(member_code)
     
    @classmethod
    def get_kosdaq_industry1_list(cls):
        return cls.obj.GetKosdaqIndustry1List()
    
    @classmethod
    def get_kosdaq_industry2_list(cls):
        return cls.obj.GetKosdaqIndustry2List()
     
    @classmethod
    def get_kosdaq_future_list(cls):
        return cls.obj.GetKostarFutureList()
    
    @classmethod
    def get_kosdaq_option_list(cls):
        return cls.obj.GetKostarOptionList()

    @classmethod
    def get_derivatives_trade_unit(cls, code):
        return cls.obj.GetFOTradeUnit(code)
    
    @classmethod
    def get_industry_group_code_list(cls, code):
        return cls.obj.GetIndustryGroupCodeList(code)

    @classmethod
    def get_index_code_list(cls, type):
        return cls.obj.GetIndexCodeList(type)
    
    @classmethod
    def get_listing_stock_number(cls, code):
        return cls.obj.GetListingStock(code)

    @classmethod
    def get_market_start_time(cls):
        return cls.obj.GetMarketStartTime()
    
    @classmethod
    def get_market_end_time(cls):
        return cls.obj.GetMarketEndTime()

    @classmethod
    def is_foreign_member(cls, member_code):
        return cls.obj.IsFrnMember(member_code)