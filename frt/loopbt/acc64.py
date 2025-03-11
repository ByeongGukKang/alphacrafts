import numba as nb
from numba.core.extending import overload_method
from numba.experimental import structref

import numpy as np


@structref.register
class Account64StructType(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)

class Account64Struct(structref.StructRefProxy):
    def __new__(
        cls,
        iter_ptr, collen, rowlen, cash, value,
        posit, posit_prc, curr_prc,
        dv_ord_queue, ord_queue, signal, mtable
    ):
        return structref.StructRefProxy.__new__(
            cls,
            iter_ptr, collen, rowlen, cash, value,
            posit, posit_prc, curr_prc,
            dv_ord_queue, ord_queue, signal, mtable
        )
    
    ### [Python (interface for numba functions)] Attributes ###

    @property
    def cash(self) -> np.float64:
        """[ ] Current cash balance of the account.
        """
        return Account64Struct_Att_cash(self)

    @property
    def cash_hist(self) -> np.ndarray:
        """[1d] Cash balance history of the account.
        """
        return Account64Struct_Att_cash_hist(self)

    @property
    def value(self) -> np.float64:
        """[ ] Current portfolio value of the account.
        """
        return Account64Struct_Att_value(self)

    @property
    def value_hist(self) -> np.ndarray:
        """[1d] Portfolio value history of the account.
        """
        return Account64Struct_Att_value_hist(self)
    
    @property
    def posit(self) -> np.ndarray:
        """[#1d] Current position of the account.
        """
        return Account64Struct_Att_posit(self)
    
    @property
    def posit_hist(self) -> np.ndarray:
        """[~2d] Position history of the account.
        """
        return Account64Struct_Att_posit_hist(self)
    
    @property
    def posit_prc(self) -> np.ndarray:
        """[#1d] Current average entry price of the account.
        """
        return Account64Struct_Att_posit_prc(self)

    ### [Python (interface for numba functions)] Methods ###

    def submit_order(self, mkey: str, orders: np.ndarray):
        Account64Struct_Met_submit_order(self, mkey, orders)

    def submit_signal(self, mkey: str, sig: np.ndarray, alloc_dv: float):
        Account64Struct_Met_submit_signal(self, mkey, sig, alloc_dv)

    def put_signal(self, signame: str, sig: np.ndarray):
        Account64Struct_Met_put_signal(self, signame, sig)
    
    def get_signal(self, signame: str) -> np.ndarray:
        """[~2d] Get memory of given signal name.
        """
        return Account64Struct_Met_get_signal(self, signame)

    def submit_signal_from(self, signame: str, alloc_dv: float):
        Account64Struct_Met_submit_signal_from(self, signame, alloc_dv)
    
    def submit_signal_decay(self, signame: str, alloc_dv: float, rate: float):
        Account64Struct_Met_submit_signal_decay(self, signame, alloc_dv, rate)
    
    def submit_signal_decay_with(self, signame: str, alloc_dv: float, rates: np.ndarray):
        Account64Struct_Met_submit_signal_decay_with(self, signame, alloc_dv, rates)
    

### Struct Type Creation ###
structref.define_proxy(
    Account64Struct,
    Account64StructType,
    [
        'iter_ptr','collen','rowlen','cash','value',
        'posit','posit_prc','curr_prc',
        'dv_ord_queue', 'ord_queue','signal', 'mtable'
    ]
)
Account64StructTypeRef = Account64StructType(
    fields=(
        ('iter_ptr', nb.int64),
        ('collen', nb.int64),
        ('rowlen', nb.int64),
        ('cash', nb.float64[::1]),
        ('value', nb.float64[::1]),
        ('posit', nb.float64[:,::1]),
        ('posit_prc', nb.float64[::1]),
        ('curr_prc', nb.float64[::1]),
        ('dv_ord_queue', nb.types.ListType(nb.float64[::1])),
        ('ord_queue', nb.types.ListType(nb.float64[::1])),
        ('signal', nb.types.DictType(nb.types.unicode_type, nb.float64[:,::1])),
        ('mtable', nb.types.DictType(nb.types.unicode_type, nb.intp[::1]))
    )
)

### Attribute Getter Functions ###
# @nb.jit(nb.int64(Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
# def Account64Struct_Att_iter_ptr(self:Account64Struct):
#     return self.iter_ptr

# @nb.jit(nb.int64(Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
# def Account64Struct_Att_collen(self:Account64Struct):
#     return self.collen

@nb.jit(nb.int64(Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
def Account64Struct_Att_rowlen(self:Account64Struct):
    return self.rowlen

@nb.jit(nb.float64[::1](Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
def Account64Struct_Att_cash_hist(self:Account64Struct):
    return self.cash

@nb.jit(nb.float64(Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
def Account64Struct_Att_cash(self:Account64Struct):
    return self.cash[self.iter_ptr]

@nb.jit(nb.float64(Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
def Account64Struct_Att_value(self:Account64Struct):
    return self.value[self.iter_ptr]

@nb.jit(nb.float64[::1](Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
def Account64Struct_Att_value_hist(self:Account64Struct):
    return self.value

@nb.jit(nb.float64[::1](Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
def Account64Struct_Att_posit(self:Account64Struct):
    return self.posit[self.iter_ptr][self.mtable['#']]

@nb.jit(nb.float64[:,::1](Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
def Account64Struct_Att_posit_hist(self:Account64Struct):
    return self.posit

@nb.jit(nb.float64[::1](Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
def Account64Struct_Att_posit_prc(self:Account64Struct):
    return self.posit_prc[self.mtable['#']]

# @nb.jit(nb.types.DictType(nb.types.unicode_type, nb.intp[::1])(Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
# def Account64Struct_Att_mtable(self:Account64Struct):
#     return self.mtable

@nb.jit(nb.intp[::1](Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
def Account64Struct_Att_mposit(self:Account64Struct):
    return self.mtable['#']

# @nb.jit(nb.intp[::1](Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
# def Account64Struct_Att_muniv(self:Account64Struct):
#     return self.mtable['@']

# @nb.jit(nb.types.ListType(nb.float64[::1])(Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
# def Account64Struct_Att_ordqueue(self:Account64Struct):
#     return self.dv_ord_queue

# @nb.jit(nb.bool_(Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
# def Account64Struct_Att_ordqueue_isfilled(self:Account64Struct):
#     if len(self.dv_ord_queue) > 0:
#         return True
#     return False

# @nb.jit(nb.float64[::1](Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
# def Account64Struct_Att_curr_prc(self:Account64Struct):
#     return self.curr_prc

# @nb.jit(nb.types.DictType(nb.types.unicode_type, nb.float64[:,::1])(Account64StructTypeRef,), nopython=True, boundscheck=False, cache=True)
# def Account64Struct_Att_signal(self:Account64Struct):
#     return self.signal


### Struct Methods ###

# submit_order
@nb.jit(
    nb.void(Account64StructTypeRef, nb.types.unicode_type, nb.float64[::1]),
    nopython=True,
    boundscheck=False,
    cache=True
)
def Account64Struct_Met_submit_order(self: Account64Struct, mkey: nb.types.unicode_type, orders: nb.float64[::1]):
    orders[np.isnan(orders)] = 0 # NaN orders are set to 0
    # Orders
    new_ord_case = np.zeros(self.collen, dtype=np.float64) 

    if mkey == '#':   # Maybe stop-loss or closing all position?
        new_ord_case[self.mtable[mkey]] = orders
    elif mkey == '@': # Maybe take-profit?
        new_ord_case[self.mtable[mkey]] = orders
    elif mkey == '~':
        new_ord_case = orders
    elif mkey == '!':
        if self.mtable.get('!') is None:
            # Lazy parsing for '!', muniv | mposit
            # Since it requires union operation, it is not efficient to calculate it every time
            self.mtable['!'] = np.union1d(self.mtable['@'], self.mtable['#']) 
        new_ord_case[self.mtable[mkey]] = orders
    else:
        raise KeyError("mkey must be one of '~', '#', '~', '!'")
    
    # Append to order queue
    self.ord_queue.append(new_ord_case)

@overload_method(Account64StructType, 'submit_order')
def ol_submit_order(self):
    return Account64Struct_Met_submit_order

# submit_signal
@nb.jit(
    nb.void(Account64StructTypeRef, nb.types.unicode_type, nb.float64[::1], nb.float64),
    nopython=True,
    boundscheck=False,
    cache=True
)
def Account64Struct_Met_submit_signal(self: Account64Struct, mkey: nb.types.unicode_type, tsig: nb.float64[::1], alloc_dv: nb.float64):
    # Check if the sum of signal is zero
    tsig[np.isnan(tsig)] = 0
    tsig_sum = np.sum(np.abs(tsig))
    if tsig_sum == 0:
        raise ValueError('sum of abs(signal) is zero')
    
    # Dollar value weight
    weight_dv = np.zeros(self.collen, dtype=np.float64) # Dollar value weight, (not_nan_tsig / tsig_sum) is the weight of each signal
    if mkey == '~':
        weight_dv = (tsig / tsig_sum) * alloc_dv 
    elif mkey == '@':
        weight_dv[self.mtable[mkey]] = (tsig / tsig_sum) * alloc_dv
    elif mkey == '#':
        weight_dv[self.mtable[mkey]] = (tsig / tsig_sum) * alloc_dv
    elif (mkey == '!') & (self.mtable.get('!') is None):
        # Lazy parsing for '!', muniv | mposit
        # Since it requires union operation, it is not efficient to calculate it every time
        self.mtable['!'] = np.union1d(self.mtable['@'], self.mtable['#']) 
        weight_dv[self.mtable[mkey]] = (tsig / tsig_sum) * alloc_dv
    else:
        raise KeyError("mkey must be one of '~', '#', '~', '!'")

    # Append to order queue
    self.dv_ord_queue.append(weight_dv) # order in dollar value

@overload_method(Account64StructType, 'submit_signal')
def ol_submit_signal(self):
    return Account64Struct_Met_submit_signal

# put_signal
@nb.jit(
    nb.void(Account64StructTypeRef, nb.types.unicode_type, nb.float64[::1]),
    nopython=True,
    boundscheck=False,
    cache=True
)
def Account64Struct_Met_put_signal(self: Account64Struct, signame: nb.types.unicode_type, new_sig: nb.float64[::1]):
    new_sig_case = np.zeros(self.collen, dtype=np.float64)
    mkey = signame[0]
    if mkey == "~":
        new_sig_case = new_sig
    elif (mkey == '!') & (self.mtable.get('!') is None):
        self.mtable['!'] = np.union1d(self.mtable['@'], self.mtable['#'])
    else:
        new_sig_case[self.mtable[mkey]] = new_sig # mkey parsing 

    sig = self.signal[signame]     # Get
    sig_maxidx = sig.shape[0]-1
    sig[:sig_maxidx] = sig[1:]     # Shift
    sig[sig_maxidx] = new_sig_case # Append new 
    self.signal[signame] = sig     # Update

@overload_method(Account64StructType, 'put_signal')
def ol_put_signal(self):
    return Account64Struct_Met_put_signal

# get_signal
@nb.jit(
    nb.float64[:,::1](Account64StructTypeRef, nb.types.unicode_type),
    nopython=True,
    boundscheck=False,
    cache=True
)
def Account64Struct_Met_get_signal(self: Account64Struct, signame: nb.types.unicode_type):
    return self.signal[signame]

@overload_method(Account64StructType, 'get_signal')
def ol_get_signal(self):
    return Account64Struct_Met_get_signal

# submit_signal_from
@nb.jit(
    nb.void(Account64StructTypeRef, nb.types.unicode_type, nb.float64),
    nopython=True,
    boundscheck=False,
    cache=True
)
def Account64Struct_Met_submit_signal_from(self: Account64Struct, signame: nb.types.unicode_type, alloc_dv: nb.float64):
    sig = self.signal[signame]
    Account64Struct_Met_submit_signal(self, '~', sig[sig.shape[0]-1], alloc_dv)

@overload_method(Account64StructType, 'submit_signal_from')
def ol_submit_signal_from(self):
    return Account64Struct_Met_submit_signal_from

# submit_signal_decay
@nb.jit(
    nb.void(Account64StructTypeRef, nb.types.unicode_type, nb.float64, nb.float64),
    nopython=True,
    boundscheck=False,
    cache=True
)
def Account64Struct_Met_submit_signal_decay(self: Account64Struct, signame: nb.types.unicode_type, alloc_dv: nb.float64, rate: nb.float64):
    sig = self.signal[signame]
    sig_maxidx = sig.shape[0]-1
    result = sig[sig_maxidx].copy()
    for i in range(1, sig_maxidx+1):
        result += sig[sig_maxidx-i] * np.power(rate, i)
    Account64Struct_Met_submit_signal(self, '~', result, alloc_dv)

@overload_method(Account64StructType, 'submit_signal_decay')
def ol_submit_signal_decay(self):
    return Account64Struct_Met_submit_signal_decay

# submit_signal_decay_with
@nb.jit(
    nb.void(Account64StructTypeRef, nb.types.unicode_type, nb.float64, nb.float64[::1]),
    nopython=True,
    boundscheck=False,
    cache=True
)
def Account64Struct_Met_submit_signal_decay_with(self: Account64Struct, signame: nb.types.unicode_type, alloc_dv: nb.float64, rates: nb.float64[::1]):
    sig = self.signal[signame]
    sig_maxidx = sig.shape[0]-1
    result = np.zeros(self.collen, dtype=np.float64)
    for i in range(min(sig_maxidx, rates.size-1)+1):
        result += sig[sig_maxidx-i] * rates[i]
    Account64Struct_Met_submit_signal(self, '~', result, alloc_dv)

@overload_method(Account64StructType, 'submit_signal_decay_with')
def ol_submit_signal_decay_with(self):
    return Account64Struct_Met_submit_signal_decay_with


### Struct Related Functions ###
# Constructor
@nb.jit(
    Account64StructTypeRef(nb.int64, nb.int64, nb.float64),
    nopython=True,
    boundscheck=False,
    cache=True
)
def newAccount64(
    rowlen: np.int64,
    collen: np.int64,
    initial_cash: np.float64
):
    cash = np.zeros(rowlen, dtype=np.float64)
    cash[0] = initial_cash
    cash[len(cash)-1] = initial_cash
    value = np.zeros(rowlen, dtype=np.float64)
    value[0] = initial_cash

    return Account64Struct(
        iter_ptr     = -1,
        collen       = collen,
        rowlen       = rowlen,
        cash         = cash,
        value        = value,
        posit        = np.zeros((rowlen, collen), dtype=np.float64),
        posit_prc    = np.zeros(collen, dtype=np.float64),
        curr_prc     = np.empty(collen, dtype=np.float64),
        dv_ord_queue = nb.typed.List.empty_list(nb.float64[::1]), # Dollar value order queue from signals, will be aggregated and converted to share order
        ord_queue    = nb.typed.List.empty_list(nb.float64[::1]), # Order (in number of shares) queue
        signal       = nb.typed.Dict.empty(nb.types.unicode_type, nb.float64[:,::1]),
        mtable       = nb.typed.Dict.empty(nb.types.unicode_type, nb.intp[::1])
    )

# Initializer
@nb.jit(
    nb.void(Account64StructTypeRef, nb.types.UnicodeCharSeq(64)[::1], nb.int64[::1]),
    nopython=True,
    boundscheck=False,
    cache=True
)
def Account64Struct_Func_init_signal(acc: Account64Struct, signames: nb.types.UnicodeCharSeq(64)[::1], signals: nb.int64[::1]):
    for signame, size in zip(signames, signals):
        acc.signal[str(signame)] = np.zeros((size, acc.collen), dtype=np.float64)
    acc.mtable['#'] = np.empty(0, dtype=np.intp)                   # mask_posit initialization
    # acc.mtable['~'] = np.arange(acc.collen, dtype=np.intp)         # mask_all initialization

# Mark-to-Market
@nb.jit(
    nb.void(Account64StructTypeRef, nb.float64[::1], nb.intp[::1]),
    nopython=True,
    boundscheck=False,
    cache=True
)
def Account64Struct_Func_mark_to_mark(acc: Account64Struct, curr_prc, mask_univ):
    # Variables
    iter_ptr  = acc.iter_ptr
    next_ptr  = iter_ptr + 1 # Next row index
    curr_cash = acc.cash[iter_ptr]

    # Update Data
    acc.curr_prc    = curr_prc
    acc.mtable['@'] = mask_univ

    # Copy the current row to the next row
    acc.cash[next_ptr]  = curr_cash
    acc.posit[next_ptr] = acc.posit[iter_ptr]

    # Mark-to-Market
    acc.value[next_ptr] = (curr_cash + np.nansum(acc.posit[iter_ptr] * curr_prc))

    # Move the iter_ptr to the next row
    acc.iter_ptr = next_ptr

# Execute Orders
@nb.jit(
    nb.void(Account64StructTypeRef, nb.float64, nb.float64),
    nopython=True,
    boundscheck=False,
    cache=True
)
def Account64Struct_Func_execute(acc: Account64Struct, fee_buy: np.float64, fee_sell: np.float64):
    # Variables
    iter_ptr = acc.iter_ptr
    curr_prc = acc.curr_prc

    ### Dollar value order aggregation
    if len(acc.dv_ord_queue) > 0:
        # Aggregate dollar value orders
        target_dv_posit = np.zeros(acc.collen, dtype=np.float64)
        for dv_orders in acc.dv_ord_queue:
            target_dv_posit += dv_orders    
        acc.dv_ord_queue.clear() # Clear the dollar value order queue

        # Only the difference between the current position and the target position is converted to the order and executed
        # (target_dv_posit / acc.curr_prc) -> target number of shares
        # (target_dv_posit / acc.curr_prc) - acc.posit[acc.iter_ptr] -> order in number of shares
        acc.ord_queue.append((target_dv_posit / curr_prc) - acc.posit[iter_ptr]) 
        
    ### Execute orders in the order queue
    # If there is no order in the order queue, return
    if len(acc.ord_queue) == 0:
        return
    
    # Execute orders in the order queue
    new_cash  = np.float64(0)
    new_posit = np.zeros(acc.collen, dtype=np.float64)
    mask_nan  = np.isnan(curr_prc)
    for orders in acc.ord_queue:
        orders[mask_nan] = 0 # Orders with NaN curr_prc are set to 0
        mask_long  = np.where(orders > 0)[0] # These masks (mask_long, mask_short)
        mask_short = np.where(orders < 0)[0] # also screen orders with NaN curr_prc (because NaN orders are 0)

        cf_from_long = np.sum(orders[mask_long] * curr_prc[mask_long])
        cf_from_short = np.sum(orders[mask_short] * curr_prc[mask_short])

        new_cash -= ((cf_from_long * fee_buy) + (cf_from_short * fee_sell))
        new_posit += orders
    acc.ord_queue.clear() # Clear the order queue 

    ## Update average entry price
    curr_posit = acc.posit[iter_ptr]
    curr_posit_prc = acc.posit_prc
    mask_same_direction = (np.sign(curr_posit) * np.sign(new_posit) == 1)
    mask_diff_direction = ~mask_same_direction

    # When new_posit is in the same direction as curr_posit, the average entry price is updated
    tmp_curr_posit = curr_posit[mask_same_direction] # Since maksed variables used multiple times,
    tmp_new_posit  = new_posit[mask_same_direction]  # it is better to store them in a temporary variable to avoid masking operation
    acc.posit_prc[mask_same_direction] = np.true_divide(
        (curr_posit_prc[mask_same_direction] * tmp_curr_posit) + (curr_prc[mask_same_direction] * tmp_new_posit),
        tmp_curr_posit + tmp_new_posit
    )

    # When new_posit is larger than curr_posit, the average entry price is updated to the new price
    acc.posit_prc[mask_diff_direction] = np.where(
        np.abs(curr_posit[mask_diff_direction]) > np.abs(new_posit[mask_diff_direction]),
        curr_posit_prc[mask_diff_direction],
        curr_prc[mask_diff_direction]
    )

    ### Update account
    acc.cash[iter_ptr] += new_cash
    acc.posit[iter_ptr] += new_posit
    acc.posit_prc[acc.posit[iter_ptr] == 0] = 0 # Set average entry price to 0 when net position is 0
    acc.mtable.clear()
    acc.mtable['#'] = np.where(acc.posit[iter_ptr] != 0)[0] # Update mask_posit, mask need to be re-calculated for intp mask