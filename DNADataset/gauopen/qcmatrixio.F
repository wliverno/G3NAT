*Deck Open_Read
      Subroutine Open_Read(Name,IU,LabFil,IVers,NLab,GVers,Title,NAtoms,
     $  NBasis,NBsUse,ICharg,Multip,NE,Len12L,Len4L,IOpCl,ICGU)
      Implicit None
C
C     This file contains low-level routines to read and write binary
C     array files as Fortran unformatted file.  All routines can
C     be called from Fortran programs and are wrapped for use in
C     Python programs, usually via the QCMatEl and QCOpMat
C     modules/classes.  The file format and contents of the header
C     records are documented in unfdat.txt.
C
C     Routines defined here, with Fortran and Python call sequences:
C
C     Call Open_Read(Name,IU,LabFil,IVers,NLab,GVers,Title,NAtoms,
C    $  NBasis,NBsUse,ICharg,Multip,NE,Len12L,Len4L,IOpCl,ICGU)
C     (iu,labfil,ivers,nlab,gvers,title,natoms,nbasis,nbsuse,icharg,
C       multip,ne,len12l,len4l,iopcl,icgu) = open_read(name)
C     Open the named binary array file for reading, and return
C     the listed scalars from the initial 2 header records.  IU receives
C     the Fortran unit number of the open file, or -1 if the open failed.
C
C     Call Open_Write(Name,IU,LabFil,GVers,Title,NAtoms,NBasis,
C    $  NBsUse,ICharg,Multip,NE,IOpCl,ICGU)
C     iu = open_write(name,labfil,gvers,title,natoms,nbasis,nbsuse,
C       icharg,multip,ne,iopcl,icgu)
C     Open the named binary array file for writing and write the
C     named scalars to the initial 2 header records.  IU receives the
C     Fortran unit number of the open file, or -1 if the open failed.
C
C     Call Close_FAF(Writing,IU)
C     close_faf (writing,iu)
C     Close the file open on Fortran unit IU, writing the END record
C     if writing
C
C     Ind = Lind2C(Check,N1,N2,I,J,JSign)
C     Ind = Lind2(Check,N1,N2,I,J,JSign)
C     Ind = Lind3C(Check,N1,N2,N3,I,J,K,JSign)
C     Ind = Lind3(Check,N1,N2,N3,I,J,K,JSign)
C     Ind = Lind4C(Check,N1,N2,N3,N4,I,J,K,L,JSign)
C     Ind = Lind4(Check,N1,N2,N3,N4,I,J,K,L,JSign)
C     Ind = Lind5C(Check,N1,N2,N3,N4,N5,I,J,K,L,M,JSign)
C     Ind = Lind5(Check,N1,N2,N3,N4,N5,I,J,K,L,M,JSign)
C     (ind,JSign) = lind2c(check,n1,n2,i,j), etc.
C
C     Return the 0-based index into a linear array given 2, 3, 4, or 5
C     indices and dimensions.  Check is True to check for indices in
C     range and return -1 if they are out of range; otherwise the
C     indices are assumed to be valid.  The C versions take c-style
C     indices (0-based and rightmost dimension and index fastest
C     running) while the others are Fortran-style (1-based and
C     leftmost dimension and index fastest running).  JSign is returned
C      +/-1 to indicate whether the upper or lower triangle was selected
C     (i.e., whether to apply a JSign flip for anti-symmetric/Hermetian
C     matrices and/or take a complex conjugate.
C     Note the all functions return a 0-based index.
C
C     Call Rd_Head(IU,NLab,NAtoms,NBasis,IAn,IAtTyp,AtmChg,C,
C    $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,NShlDB,
C    $  NPrmDB,NBTot)
C     (ian,iattyp,atmchg,c,ibfatm,ibftyp,atmwgt,nfc,nfv,itran,idum9,
C       nshlao,nprmao,nshldb,nprmdb,nbtot) = rd_head(iu,nlab,natoms,nbasis)
C     Call Rd_HeadA(IU,NLab,NAtoms,NBasis,IAn,IAtTyp,AtmChg,C,
C    $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,Rec11)
C     (ian,iattyp,atmchg,c,ibfatm,ibftyp,atmwgt,nfc,nfv,itran,idum9,
C       nshlao,nprmao,nshldb,nprmdb,nbtot) = rd_head(iu,nlab,natoms,nbasis)
C     (ian,iattyp,atmchg,c,ibfatm,ibftyp,atmwgt,nfc,nfv,itran,idum9,
C       rec11) = rd_heada(iu,nlab,natoms,nbasis)
C     Read the remaining header records (3 to NLab) and return the named
C     arrays.  Rd_HeadA returns the full 16 integer array in record 11.
C
C     Call Rd_Labl(IU,IVers,CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,TypeA,
C    $  NRI,EOF)
C     (cbuf,ni,nr,ntot,lenbuf,n1,n2,n3,n4,n5,typea,nri,eof) = rd_labl(iu,ivers)
C     read the label record for an operator matrix.
C
C     Call Rd_IBuf(IU,NI*NTot,NR*LenBuf,Arr)
C     arr = rd_ibuf(iu,ni*ntot,ni*lenbuf)
C     Call Rd_RBuf(IU,NR*NTot,NR*LenBuf,Arr)
C     arr = rd_rbuf(iu,nr*ntot,nr*lenbuf)
C     Call Rd_CBuf(IU,NR*NTot,NR*LenBuf,Arr)
C     arr = rd_cbuf(iu,nr*ntot,nr*lenbuf)
C     read an integer/real/complex array from the file given parameters
C     from the header record.
C
C     Call Rd_RInd(IU,NR,LR,NTot,LenBuf,RArr)
C     arr = rd_rind(iu,nr,lr,ntot,lenbuf)
C     Read a real array stored with indices for non-zero elements.
C
C     Call Rd_Skip(IU,NTot,LenBuf)
C     rd_skip(iu,ntot,lenbuf)
C     Skip the data records for an object on the file having NTot
C     elements stored with LenBuf per record.
C
C     Call Rd_2E1(IU,LR,NTot,LenBuf,RArr)
C     Call Rd_2EN(IU,NR,LR,LRNR,NTot,LenBuf,RArr)
C     arr = rd_2e1(iu,lr,ntot,lenbuf)
C     arr = rd_2en(iu,nr,lr,lrnr,ntot,lenbuf)
C     Read and return an array of AO 2e integrals (really, a
C     4-dimensional array with quartets of indices and one
C     value or NR values per index set).  NTot is the number of non-zero
C     values (from the header record for the object) and
C     LR is the total number of elements (LenArr(-N,-N,-N,N,1), where
C     N is the number of basis functions).
C
C     Call Rd_SpA(IU,NI,NR,NTot,LenBuf,IArr,RArr)
C     Call Rd_SpAC(IU,NI,NR,NTot,LenBuf,IArr,RArr)
C     iarr,rarr = rd_spa(IU,NI,NR,NTot,LenBuf)
C     iarr,carr = rd_spac(IU,NI,NR,NTot,LenBuf)
C     Read and return a list of values and indices for a real or complex
C     array which is kept in sparse form.  NR is the number of values,
C     NI is the number of indices and NTot is the number of value
C     sets.
C
C     Call Wr_Head(IU,NAtoms,NAt3,NBasis,IAn,IAtTyp,AtmChg,C,
C    $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,NShlDB,
C    $  NPrmDB,NBTot)
C     Call Wr_HeadA(IU,NAtoms,NAt3,NBasis,IAn,IAtTyp,AtmChg,C,
C    $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,Rec11)
C     Call Wr_DummyHead(IU)
C     wr_head(iu,natoms,nat3,nbasis,ian,iattyp,atmchg,c,ibfatm,ibftyp,
C       atmwgt,nfc,nfv,itran,idum9,nshlao,nprmao,nshldb,nprmdb,nbtot)
C     wr_head(iu,natoms,nat3,nbasis,ian,iattyp,atmchg,c,ibfatm,ibftyp,
C       atmwgt,nfc,nfv,itran,idum9,rec11)
C     wr_dummyhead(iu)
C     Write the header records (3 to NLab) to Fortran unit IU.  Wr_HeadA
C     expects the full 16 integer array for record 11.  Wr_DummyHead
C     writes a header with dummy information, for a file which is just
C     used for arrays and does not need a molecule specification.
C
C     The following low-level routines are not usually called directly,
C     but are accessed via the routine Wr_L{IBuf,RBuf,CBuf,RInd} which
C     are in qcmatrix.F for fortran and QCMatEl.py for Python.
C
C     Call Wr_Labl(IU,CBuf,NI,NR,NTot,LenBfX,N1,N2,N3,N4,N5,TypeA)
C     Write the header record for one matrix to the file.
C
C     Call Wr_IBuf(IU,NI*NTot,LenBfX,Arr)
C     Call Wr_RBuf(IU,NR*NTot,LenBfX,Arr)
C     Call Wr_CBuf(IU,NR*NTot,LenBfX,Arr)
C     Call Wr_RInd(IU,NR,LR,NTot,LenBfX,RArr)
C     Call Wr_2E(IU,NTot,NR,N,LR,LenBfX,RArr)
C     Call Wr_SpA(IU,NI,NR,NTot,LenBuf,IArr,RArr)
C     Write objects of the specified types to the file.  LenBfX is
C     derived from the general LenBuf based on the length of the
C     items; this is normally handled by the Wr_Lxxxx routines.
C
C     LVal = AOInts(CBuf)
C     lval = aoints(cbuf)
C     Return true if the operator identified by the string in CBuf
C     is an AO 2e integral array (regular or Rafenetti).
C
C     LVal = DAOInts(CBuf)
C     lval = saoints(cbuf)
C     Return true if the operator identified by the string in CBuf
C     is an AO 2e integral derivative array.
C
C     NTot = LenArr(N1,N2,N3,N4,N5)
C     ntot = lenarr(n1,n2,n3,n4,n5)
C     return the total number of index values of an array with
C     the specified dimensions, accounting for possible
C     lower-triangular indices.  This does not include a
C     possible multiple number of values per index (NI or NR in
C     the record header, nelem in the object).
C
C     NNZ = QCM_NumNZR(NR,NTot,X)
C     return the number of non-zero elements of X(NR,NTot) (Fortran order).
C     Returns at least 1 so that something will be written even for zero
C     arrays.
C
C     MaxI = QCM_MaxINZR(NR,NTot,X)
C     Return the (1-based) index into the last non-zero column of X,
C     or 0 if all columns are zero.
C
C     Open a Gaussian binary array file and read header information.
C     IU receives the Fortran unit number or -1 if the open failed.
C
      Integer LStr, IUUse, Len12D, Len4D, IUSt, IUEnd
#ifdef USE_I8
      Parameter (Len12D=8,Len4D=8)
#else
      Parameter (Len12D=4,Len4D=4)
#endif
      Parameter (LStr=64,IUSt=57,IUEnd=99)
C     The latest f2py is now "improved" and is now too stupid to handle
C     character string lengths given by parameters.
C     Character*(*) Name, LabFil*(LStr), GVers*(LStr), Title*(LStr)
      Character*(*) Name, LabFil*64, GVers*64, Title*64
      Logical IsOpen
      Integer IU,IVers,NLab,NAtoms,NBasis,NBsUse,ICharg,Multip,NE,
     $  Len12L,Len4L,IOpCl,ICGU
cf2py intent(out) labfil,gvers,title,iu,ivers,nlab,natoms,nbasis,nbsuse
cf2py intent(out) icharg,multip,ne,len12l,len4l,iopcl,icgu
 1000 Format(' This QCMatrixIO was compiled with Len12=',I1,' Len4=',I1,
     $  ' but file has Len12=',I1,' Len4=',I1,'.')
C
      LabFil = ' '
      IVers = 0
      NLab = 0
      GVers = ' '
      Title = ' '
      NAtoms = 0
      NBasis = 0
      NBsUse = 0
      ICharg = 0
      Multip = 0
      NE = 0
      Len12L = 0
      Len4L = 0
      IOpCl = 0
      ICGU = -1
      Do 10 IUUse = IUSt, IUEnd
        Inquire(Unit=IUUse,Opened=IsOpen)
        If(.not.IsOpen) goto 20
   10   Continue
      IU = -1
      Return
C
   20 Open (Unit=IUUse,File=Name,Form='Unformatted',Status='Old',
     $  Err=900)
      IU = IUUse
      Read(IU) LabFil(1:LStr), IVers, NLab, GVers(1:LStr)
      If(Len4D.eq.4.and.IVers.le.0.or.NLab.le.0.or.NLab.gt.10000) then
C       Looks like an 8-byte integer file but we are using 4-byte ints.
        IU = -2
      else if(IVers.eq.1) then
        Read(IU) Title(1:LStr), NAtoms, NBasis, NBsUse, ICharg, Multip,
     $    NE, Len12L, Len4L
      else
        Read(IU) Title(1:LStr), NAtoms, NBasis, NBsUse, ICharg, Multip,
     $    NE, Len12L, Len4L, IOpCl, ICGU
        endIf
      If(Len4L.ne.Len4D.or.Len12L.ne.Len12D) then
        If(IU.ne.-2) Write(6,1000) Len12D, Len4D, Len12L, Len4L
        Close(IUUse)
        endIf
  900 Return
      End
*Deck Open_Write
      Subroutine Open_Write(Name,IU,LabFil,GVers,Title,NAtoms,NBasis,
     $  NBsUse,ICharg,Multip,NE,IOpCl,ICGU)
      Implicit None
C
C     Open a Gaussian binary array file and write header information.
C     IU receives the Fortran unit number or -1 if the open failed.
C
      Integer LStr,IUUse,IVers,Len12L,Len4L,NLab,IUSt,IUEnd
#ifdef USE_I8
      Parameter (Len12L=8,Len4L=8)
#else
      Parameter (Len12L=4,Len4L=4)
#endif
      Parameter (LStr=64,IUSt=57,IUEnd=99,IVers=3,NLab=11)
C     The latest f2py is now "improved" and is now too stupid to handle
C     character string lengths given by parameters.
C     Character*(*) Name, LabFil, GVers, Title, LLabFil*(LStr),
C    $  LGVers*(LStr), LTitle*(LStr)
      Logical IsOpen
      Character*(*) Name, LabFil, GVers, Title, LLabFil*64,
     $  LGVers*64, LTitle*64
      Integer IU,NAtoms,NBasis,NBsUse,ICharg,Multip,NE,IOpCl,ICGU
CF2PY Intent(Out) IU
C
      Do 10 IUUse = IUSt, IUEnd
        Inquire(Unit=IUUse,Opened=IsOpen)
        If(.not.IsOpen) goto 20
   10   Continue
      IU = -1
      Return
C
   20 Open (Unit=IUUse,File=Name,Form='Unformatted',Status='Unknown',
     $  Err=900)
      IU = IUUse
      LLabFil = LabFil
      LGVers = GVers
      LTitle = Title
      Write(IU) LLabFil, IVers, NLab, LGVers
      Write(IU) LTitle, NAtoms, NBasis, NBsUse, ICharg, Multip, NE,
     $  Len12L, Len4L, IOpCl, ICGU
      Return
C
  900 IU = -1
      Return
      End
*Deck Close_FAF
      Subroutine Close_FAF(DoWrt,IU)
      Implicit None
C
C     Close a binary array file.
C
      Integer IU, I0
      Parameter (I0=0)
      Logical DoWrt
C
      If(DoWrt) Call Wr_Labl(IU,'END',I0,I0,I0,I0,I0,I0,I0,I0,I0,I0)
      Close (Unit=IU)
      Return
      End
*Deck AOInts
      Logical Function AOInts(CBuf)
      Implicit None
      Character*(*) CBuf, Reg, Raf
      Parameter (Reg='REGULAR 2E INTEGRALS',
     $  Raf='RAFFENETTI 2E INTEGRALS')
C
      AOInts = CBuf.eq.Reg.or.CBuf.eq.Raf
      Return
      End
*Deck DAOInts
      Logical Function DAOInts(CBuf)
      Implicit None
      Character*(*) CBuf, Reg
      Parameter (Reg='2E AO INTEGRAL NUCLEAR DERIVATIVES')
C
      DAOInts = CBuf.eq.Reg
      Return
      End
*Deck QCM_AOInts
      Logical Function QCM_AOInts(CBuf)
      Implicit None
      Character*(*) CBuf, Reg, Raf
      Parameter (Reg='REGULAR 2E INTEGRALS',
     $  Raf='RAFFENETTI 2E INTEGRALS')
C
      QCM_AOInts = CBuf.eq.Reg.or.CBuf.eq.Raf
      Return
      End
*Deck QCM_DAOInts
      Logical Function QCM_DAOInts(CBuf)
      Implicit None
      Character*(*) CBuf, Reg
      Parameter (Reg='2E AO INTEGRAL NUCLEAR DERIVATIVES')
C
      QCM_DAOInts = CBuf.eq.Reg
      Return
      End
*Deck LenArr
      Integer Function LenArr(N1,N2,N3,N4,N5)
      Implicit None
      Integer N1,N2,N3,N4,N5,QCM_LenArray
C
      LenArr = QCM_LenArray(N1,N2,N3,N4,N5)
      Return
      End
*Deck QCM_LenArray
      Integer Function QCM_LenArray(N1,N2,N3,N4,N5)
      Implicit None
      Integer N1,N2,N3,N4,N5,N1X,N2X,N3X,N4X,N5X,Abs,QCM_Lind5,JSign
C
      N1X = N1
      If(N1X.eq.0) N1X = 1
      N2X = N2
      If(N2X.eq.0) N2X = 1
      N3X = N3
      If(N3X.eq.0) N3X = 1
      N4X = N4
      If(N4X.eq.0) N4X = 1
      N5X = N5
      If(N5X.eq.0) N5X = 1
      QCM_LenArray = QCM_Lind5(N1X,N2X,N3X,N4X,N5X,Abs(N1X),
     $  Abs(N2X),Abs(N3X),Abs(N4X),Abs(N5X),JSign) + 1
      Return
      End
*Deck LInd2C
      Integer Function LInd2C(Check,N1,N2,I,J,JSign)
      Implicit None
C
C     Linear or square indexing, I,J are 0-based, c order
C     output is 0-based.  JSign is +/-1.
C
      Logical Check
      Integer N1,N2,I,J,Lind2,JSign
CF2PY Intent (Out) JSign
C
      Lind2C = Lind2(Check,N2,N1,J+1,I+1,JSign)
      Return
      End
*Deck LInd2
      Integer Function Lind2(Check,N1,N2,I,J,JSign)
      Implicit None
C
C     Linear or square indexing, I,J are 1-based,
C     output is 0-based.  JSign is +/-1.
C
      Logical Check
      Integer N1,N2,I,J,JSign,QCM_Lind2
CF2PY Intent (Out) JSign
C
      If(Check.and.(N2.le.0.or.N1.eq.0.or.(N1.lt.0.and.N1.ne.(-N2)).or.
     $  I.lt.1.or.I.gt.Abs(N1).or.J.lt.1.or.J.gt.N2)) then
        JSign = 1
        Lind2 = -1
      else
        Lind2 = QCM_Lind2(N1,N2,I,J,JSign)
        endIf
      Return
      End
*Deck QCM_LInd2
      Integer Function QCM_Lind2(N1,N2,I,J,JSign)
      Implicit None
C
C     Linear or square indexing, I,J are 1-based,
C     output is 0-based.  JSign is +/-1.
C
      Integer N1,N2,I,J,JSign
CF2PY Intent (Out) JSign
C
      JSign = 1
      If(N1.lt.0) then
        If(I.ge.J) then
          QCM_Lind2 = (I*(I-1))/2 + J - 1
        else
          QCM_Lind2 = (J*(J-1))/2 + I - 1
          JSign = -1
          endIf
      else
        QCM_Lind2 = N1*(J-1) + I - 1
        endIf
      Return
      End
*Deck Lind3C
      Integer Function Lind3C(Check,N1,N2,N3,I,J,K,JSign)
      Implicit None
C
C     Linear or square indexing, I,J,K are 0-based, c order
C     output is 0-based.  JSign is +/-1.
C
      Logical Check
      Integer N1,N2,N3,I,J,K,Lind3,JSign
CF2PY Intent (Out) JSign
C
      Lind3C = Lind3(Check,N3,N2,N1,K+1,J+1,I+1,JSign)
      Return
      End
*Deck LInd3
      Integer Function Lind3(Check,N1,N2,N3,I,J,K,JSign)
      Implicit None
C
C     Linear or square indexing, I,J,K are 1-based,
C     output is 0-based.  JSign is +/-1.
C
      Logical Check
      Integer N1,N2,N3,I,J,K,QCM_Lind3,JSign
CF2PY Intent (Out) JSign
C
      If(Check.and.(N3.le.0.or.(N1*N2).eq.0.or.
     $  (N1.lt.0.and.N1.ne.(-Abs(N2))).or.
     $  (N2.lt.0.and.N2.ne.(-N3)).or.I.lt.1.or.I.gt.Abs(N1).or.
     $  J.lt.1.or.J.gt.Abs(N2).or.K.lt.1.or.K.gt.N3)) then
        JSign = 1
        Lind3 = -1
      else
        Lind3 = QCM_Lind3(N1,N2,N3,I,J,K,JSign)
        endIf
      Return
      End
*Deck QCM_LInd3
      Integer Function QCM_Lind3(N1,N2,N3,I,J,K,JSign)
      Implicit None
C
C     Linear or square indexing, I,J,K are 1-based,
C     output is 0-based.  JSign is +/-1.
C
      Integer N1,N2,N3,I,J,K,I1,J1,K1,N12,IJ,QCM_Lind2,JSign
CF2PY Intent (Out) JSign
C
      JSign = 1
      I1 = I - 1
      J1 = J - 1
      K1 = K - 1
      If(N1.gt.0) then
        If(N2.gt.0) then
          QCM_Lind3 = N1*(N2*K1+J1) + I1
        else
          QCM_Lind3 = N1*QCM_Lind2(N2,N3,J,K,JSign) + I1
          endIf
      else if(N2.gt.0) then
        N12 = (N2*(N2+1))/2
        If(I.ge.J) then
          IJ = (I*I1)/2 + J1
        else
          IJ = (J*J1)/2 + I1
          JSign = -1
          endIf
        QCM_Lind3 = N12*K1 + IJ
      else
        K1 = Max(I,J,K) - 1
        I1 = Min(I,J,K) - 1
        J1 = I + J + K - K1 - I1 - 3
        QCM_Lind3 = I1 + (J1*(J1+1))/2 + (K1*(K1+1)*(K1+2))/6
        endIf
      Return
      End
*Deck Lind4C
      Integer Function Lind4C(Check,N1,N2,N3,N4,I,J,K,L,JSign)
      Implicit None
C
C     Linear or square indexing, I,J,K,L are 0-based, c order
C     output is 0-based.  JSign is +/-1.
C
      Logical Check
      Integer N1,N2,N3,N4,I,J,K,L,Lind4,JSign
CF2PY Intent (Out) JSign
C
      Lind4C = Lind4(Check,N4,N3,N2,N1,L+1,K+1,J+1,I+1,JSign)
      Return
      End
*Deck LInd4
      Integer Function Lind4(Check,N1,N2,N3,N4,I,J,K,L,JSign)
      Implicit None
C
C     Linear or square indexing, I,J,K,L are 1-based,
C     output is 0-based.  JSign is +/-1.
C
      Logical Check
      Integer N1,N2,N3,N4,I,J,K,L,QCM_Lind4,JSign
CF2PY Intent (Out) JSign
C
      If(Check.and.(N4.le.0.or.(N1*N2*N3).eq.0.or.
     $  (N1.lt.0.and.N1.ne.(-Abs(N2))).or.
     $  (N2.lt.0.and.N2.ne.(-Abs(N3))).or.
     $  (N3.lt.0.and.N3.ne.(-N4)).or.
     $  I.lt.1.or.I.gt.Abs(N1).or.J.lt.1.or.J.gt.Abs(N2).or.
     $  K.lt.1.or.K.gt.Abs(N3).or.L.lt.1.or.L.gt.N4)) then
        Lind4 = -1
        JSign = 1
      else
        Lind4 = QCM_Lind4(N1,N2,N3,N4,I,J,K,L,JSign)
        endIf
      Return
      End
*Deck QCM_LInd4
      Integer Function QCM_Lind4(N1,N2,N3,N4,I,J,K,L,JSign)
      Implicit None
C
C     Linear or square indexing, I,J,K,L are 1-based,
C     output is 0-based.  JSign is +/-1.
C
      Integer N1,N2,N3,N4,I,J,K,L,I1,J1,K1,L1,ICase,QCM_Lind2,N23,
     $  QCM_Lind3,N123,KL,IJ,IJK,JKL,JK,N12,JSign,JSignIJ,JSignKL
CF2PY Intent (Out) JSign
C
      I1 = I - 1
      J1 = J - 1
      K1 = K - 1
      L1 = L - 1
      ICase = 0
      If(N1.lt.0) ICase = ICase + 1
      If(N2.lt.0) ICase = ICase + 2
      If(N3.lt.0) ICase = ICase + 4
      Goto (100,110,120,130,140,150,160,170), ICase+1
C
C     No symmetries.
  100 QCM_Lind4 = N1*(N2*(N3*L1+K1)+J1) + I1
      JSign = 1
      Return
C
C     I<=J
  110 IJ = QCM_Lind2(N1,N2,I,J,JSign)
      N12 = (N2*(N2+1))/2
      QCM_Lind4 = N12*(N3*L1+K1) + IJ
      Return
C
C     I,J<=K,L
  120 JK = QCM_Lind2(N2,N3,J,K,JSign)
      N23 = (N3*(N3+1))/2
      QCM_Lind4 = N1*(N23*L1+JK) + I1
      Return
C
C     I<=J<=K,L
  130 IJK = QCM_Lind3(N1,N2,N3,I,J,K,JSign)
      N123 = (N3*(N3+1)*(N3+2))/6
      QCM_Lind4 = N123*L1 + IJK
      Return
C
C     I,J,K<=L
  140 KL = QCM_Lind2(N3,N4,K,L,JSign)
      QCM_Lind4 = N1*(N2*KL+J1) + I1
      Return
C
C     I<=J,K<=L
  150 IJ = QCM_Lind2(N1,N2,I,J,JSignIJ)
      KL = QCM_Lind2(N3,N4,K,L,JSignKL)
      N12 = (N2*(N2+1))/2
      QCM_Lind4 = N12*KL + IJ
      JSign = JSignIJ*JSignKL
      Return
C
C     I,J<=K<=L
  160 JKL = QCM_Lind3(N2,N3,N4,J,K,L,JSign)
      QCM_Lind4 = N1*JKL + I1
      Return
C
C     I<=J<=K<=L
  170 IJ = QCM_Lind2(N1,N4,I,J,JSignIJ)
      KL = QCM_Lind2(N3,N4,K,L,JSignKL)
      QCM_Lind4 = QCM_Lind2(N1,N4,IJ+1,KL+1,JSign)
      JSign = JSign*JSignIJ*JSignKL
      Return
      End
*Deck Lind5C
      Integer Function Lind5C(Check,N1,N2,N3,N4,N5,I,J,K,L,M,JSign)
      Implicit None
C
C     Linear or square indexing, I,J,K,L,M are 0-based, c order
C     output is 0-based.  JSign is +/-1.
C
      Logical Check
      Integer N1,N2,N3,N4,N5,I,J,K,L,M,Lind5,JSign
CF2PY Intent (Out) JSign
C
      Lind5C = Lind5(Check,N5,N4,N3,N2,N1,M+1,L+1,K+1,J+1,I+1,JSign)
      Return
      End
*Deck LInd5
      Integer Function Lind5(Check,N1,N2,N3,N4,N5,I,J,K,L,M,JSign)
      Implicit None
C
C     Linear or square indexing, I,J,K,L,M are 1-based, output is
C     0-based.  JSign is +/-1, LM indices can not be lower triangular.
C
      Logical Check
      Integer N1,N2,N3,N4,N5,I,J,K,L,M,JSign,QCM_Lind5
CF2PY Intent (Out) JSign
C
      If(Check.and.(N5.le.0.or.N4.le.0.or.(N1*N2*N3).eq.0.or.
     $  (N1.lt.0.and.N1.ne.(-Abs(N2))).or.
     $  (N2.lt.0.and.N2.ne.(-Abs(N3))).or.
     $  (N3.lt.0.and.N3.ne.(-N4)).or.
     $  I.lt.1.or.I.gt.Abs(N1).or.J.lt.1.or.J.gt.Abs(N2).or.
     $  K.lt.1.or.K.gt.Abs(N3).or.L.lt.1.or.L.gt.N4.or.
     $  M.lt.1.or.M.gt.N5)) then
        Lind5 = -1
        JSign = 1
      else
        LInd5 = QCM_LInd5(N1,N2,N3,N4,N5,I,J,K,L,M,JSign)
        endIf
      Return
      End
*Deck QCM_LInd5
      Integer Function QCM_LInd5(N1,N2,N3,N4,N5,I,J,K,L,M,JSign)
      Implicit None
C
C     Linear or square indexing, I,J,K,L,M are 1-based, output is
C     0-based.  JSign is +/-1, LM indices can not be lower triangular.
C
      Integer N1,N2,N3,N4,N5,I,J,K,L,M,M1,QCM_Lind4,N1A,N2A,N3A,N1234,
     $  JSign
CF2PY Intent (Out) JSign
C
      M1 = M - 1
      N1A = Abs(N1)
      N2A = Abs(N2)
      N3A = Abs(N3)
      N1234 = QCM_Lind4(N1,N2,N3,N4,N1A,N2A,N3A,N4,JSign)+1
      QCM_LInd5 = QCM_Lind4(N1,N2,N3,N4,I,J,K,L,JSign)+N1234*M1
      Return
      End
*Deck QCM_MaxINZR
      Integer Function QCM_MaxINZR(NR,LR,X)
      Implicit None
      Integer NR,LR,I,J,I1
      Real*8 X(NR,LR),Zero
      Parameter (Zero=0.0d0)
C
      QCM_MaxINZR = 0
      Do 20 I = 1, LR
        I1 = 0
        Do 10 J = 1, NR
          If(X(J,I).ne.Zero) I1 = 1
   10     Continue
   20   If(I1.eq.1) QCM_MaxINZR = I
      Return
      End
*Deck QCM_NumNZR
      Integer Function QCM_NumNZR(NR,LR,X)
C     Calculates the number of non-zero sets of nr elements in x (checks
C     for any non-zero value in a set of nr values). Assumes LR is the 
C     slowest running values
      Implicit None
      Integer NR,LR,I,J,I1
      Real*8 X(NR,LR),Zero
      Parameter (Zero=0.0d0)
C
      QCM_NumNZR = 0
      Do 20 I = 1, LR
        I1 = 0
        Do 10 J = 1, NR
          If(X(J,I).ne.Zero) I1 = 1
   10     Continue
   20   QCM_NumNZR = QCM_NumNZR + I1

C     If no non-zero indexed values, return at least 1
      If(QCM_NumNZR.eq.0) QCM_NumNZR = 1
      Return
      End
*Deck QCM_NumNZT
      Integer Function QCM_NumNZT(NR,LR,X)
C     Calculates the number of non-zero sets of nr elements in x (checks
C     for any non-zero value in a set of nr values). Assumes nr is the 
C     slowest running values

      Implicit None
      Integer NR,LR,I,J,I1
      Real*8 X(LR,NR),Zero
      Parameter (Zero=0.0d0)
C
      QCM_NumNZT = 0
      Do 20 I = 1, LR
        I1 = 0
        Do 10 J = 1, NR
          If(X(I,J).ne.Zero) I1 = 1
   10     Continue
   20   QCM_NumNZT = QCM_NumNZT + I1

C     If no non-zero indexed values, return at least 1
      If(QCM_NumNZT.eq.0) QCM_NumNZT = 1
      Return
      End
*Deck QCM_NumNZRP
      Integer Function QCM_NumNZRP(NR,LR,X)
C     Wrapper for QCM_NumNZR to allow X to be provided as flat array
C     (for Python interface mainly)
      Implicit None
      Integer NR,LR,QCM_NumNZR
      Real*8 X(NR*LR)
      QCM_NumNZRP = QCM_NumNZR(NR,LR,X)
      Return
      End
*Deck QCM_NumNZTP
      Integer Function QCM_NumNZTP(NR,LR,X)
C     Wrapper for QCM_NumNZT to allow X to be provided as flat array
C     (for Python interface mainly)
      Implicit None
      Integer NR,LR,QCM_NumNZT
      Real*8 X(NR*LR)
      QCM_NumNZTP = QCM_NumNZT(NR,LR,X)
      Return
      End
*Deck Rd_2E1
      Subroutine Rd_2E1(IU,LR,NTot,LenBuf,RArr)
      Implicit None
      Integer IU,LR,NTot,LenBuf,IBuf(4,LenBuf),I,Ind,NDo,IJ,KL,IJKL
      Real*8 Buf(LenBuf),RArr(LR),Zero
      Parameter (Zero=0.0d0)
CF2PY Intent (Out) RArr
C
      Do 20 I = 1, LR
   20   RArr(I) = Zero
      Do 50 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Read(IU) IBuf, Buf
        Do 40 I = 1, NDo
          If(IBuf(1,I).ge.IBuf(2,I)) then
            IJ = (IBuf(1,I)*(IBuf(1,I)-1))/2 + IBuf(2,I)
          else
            IJ = (IBuf(2,I)*(IBuf(2,I)-1))/2 + IBuf(1,I)
            endIf
          If(IBuf(3,I).ge.IBuf(4,I)) then
            KL = (IBuf(3,I)*(IBuf(3,I)-1))/2 + IBuf(4,I)
          else
            KL = (IBuf(4,I)*(IBuf(4,I)-1))/2 + IBuf(3,I)
            endIf
          If(IJ.ge.KL) then
            IJKL = (IJ*(IJ-1))/2 + KL
          else
            IJKL = (KL*(KL-1))/2 + IJ
            endIf
   40     RArr(IJKL) = Buf(I)
   50   Continue
      Return
      End
*Deck Rd_2EN
      Subroutine Rd_2EN(IU,NR,LR,LRNR,NTot,LenBuf,RArr)
      Implicit None
      Integer IU,NR,LR,NTot,LenBuf,IBuf(4,LenBuf),I,J,Ind,NDo,IJ,KL,
     $  IJKL,LRNR
      Real*8 Buf(NR,LenBuf),RArr(LRNR),Zero
      Parameter (Zero=0.0d0)
CF2PY Intent (Out) RArr
C
      Do 10 I = 1, LRNR
   10   RArr(I) = Zero
      Do 50 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Read(IU) IBuf, Buf
        Do 40 I = 1, NDo
          If(IBuf(1,I).ge.IBuf(2,I)) then
            IJ = (IBuf(1,I)*(IBuf(1,I)-1))/2 + IBuf(2,I)
          else
            IJ = (IBuf(2,I)*(IBuf(2,I)-1))/2 + IBuf(1,I)
            endIf
          If(IBuf(3,I).ge.IBuf(4,I)) then
            KL = (IBuf(3,I)*(IBuf(3,I)-1))/2 + IBuf(4,I)
          else
            KL = (IBuf(4,I)*(IBuf(4,I)-1))/2 + IBuf(3,I)
            endIf
          If(IJ.ge.KL) then
            IJKL = (IJ*(IJ-1))/2 + KL
          else
            IJKL = (KL*(KL-1))/2 + IJ
            endIf
          Do 30 J = 1, NR
   30       RArr(IJKL+(J-1)*LR) = Buf(J,I)
   40     Continue
   50   Continue
      Return
      End
*Deck Rd_SpA
      Subroutine Rd_SpA(IU,NI,NR,NTot,LenBuf,IArr,RArr)
      Implicit None
      Integer IU,NI,NR,NTot,LenBuf,IBuf(NI,LenBuf),I,J,Ind,NDo,
     $  IArr(NI*NTot),IndI,IndR
      Real*8 Buf(NR,LenBuf),RArr(NR*NTot)
CF2PY Intent (Out) IArr, RArr
C
      Do 70 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Read(IU) IBuf, Buf
        IndI = Ind*NI
        IndR = Ind*NR
        Do 60 I = 1, NDo
          Do 40 J = 1, NI
   40       IArr(J+IndI) = IBuf(J,I)
          IndI = IndI + NI
          Do 50 J = 1, NR
   50       RArr(J+IndR) = Buf(J,I)
   60     IndR = IndR + NR
   70   Continue
      Return
      End
*Deck Rd_SpAC
      Subroutine Rd_SpAC (IU,NI,NR,NTot,LenBuf,IArr,RArr)
      Implicit None
      Integer IU,NI,NR,NTot,LenBuf,IBuf(NI,LenBuf),I,J,Ind,NDo,
     $  IArr(NI*NTot),IndI,IndR
      Complex*16 Buf(NR,LenBuf),RArr(NR*NTot)
CF2PY Intent (Out) IArr, RArr
C
      Do 70 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Read(IU) IBuf, Buf
        IndI = Ind*NI
        IndR = Ind*NR
        Do 60 I = 1, NDo
          Do 40 J = 1, NI
   40       IArr(J+IndI) = IBuf(J,I)
          IndI = IndI + NI
          Do 50 J = 1, NR
   50       RArr(J+IndR) = Buf(J,I)
   60     IndR = IndR + NR
   70   Continue
      Return
      End
*Deck Rd_CBuf
      Subroutine Rd_CBuf(IU,LR,LenBuf,Arr)
      Implicit None
      Integer IU,LR,LenBuf,Ind,NDo,I
      Complex*16 Buf(LenBuf),Arr(LR)
CF2PY Intent (Out) Arr
C
      Do 20 Ind = 0, (LR-1), LenBuf
        NDo = Min(LenBuf,LR-Ind)
        Read(IU) Buf
        Do 10 I = 1, NDo
   10     Arr(Ind+I) = Buf(I)
   20   Continue
      Return
      End
*Deck Rd_ChBuf
      Subroutine Rd_ChBuf(IU,LR,LenBuf,CArr)
      Implicit None
      Integer IU,LR,LenBuf,Ind,NDo,LCArr,Lim,LimI
      Character CArr*(*), Buf*(LenBuf)
C
      LCArr = Len(CArr)
      CArr = ' '
      Do 20 Ind = 0, (LR-1), LenBuf
        NDo = Min(LenBuf,LR-Ind)
        Read(IU) Buf
        Lim = Min(Ind+NDo,LCArr)
        LimI = Lim - Ind
        If(Ind.lt.Lim) CArr(Ind+1:Lim) = Buf(1:LimI)
   20   Continue
      If(LR.lt.LCArr) CArr(LR+1:) = ' '
      Return
      End
*Deck Rd_ChBufI
      Subroutine Rd_ChBufI(IU,LR,LenBuf,CArr,IArr)
      Implicit None
      Integer IU,LR,LenBuf,IArr(LR),LCopy,I
      Character*(*) CArr
CF2PY Intent (Out) IArr
C
      Call Rd_ChBuf(IU,LR,LenBuf,CArr)
      LCopy = Min(LR,Len(CArr))
      Do 10 I = 1, LCopy
   10   IArr(I) = IChar(CArr(I:I))
      Do 20 I = (LCopy+1), LR
   20   IArr(I) = IChar(' ')
      Return
      End
*Deck Rd_Head
      Subroutine Rd_Head(IU,NLab,NAtoms,NBasis,IAn,IAtTyp,AtmChg,C,
     $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,NShlDB,
     $  NPrmDB,NBTot)
      Implicit None
      Integer IU,NLab,NAtoms,NBasis,IAn(NAtoms),IAtTyp(NAtoms),
     $  IBfAtm(NBasis),IBfTyp(NBasis),NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,
     $  NShlDB,NPrmDB,NBTot,I
      Real*8 AtmChg(NAtoms),C(3*NAtoms),AtmWgt(NAtoms)
CF2PY Intent(Out) IAn,IAtTyp,AtmChg,C,IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran
CF2PY Intent(Out) IDum9,NShlAO,NPrmAO,NShlDB,NPrmDB,NBTot
C
      NFC = 0
      NFV = 0
      ITran = 0
      IDum9 = 0
      NShlAO = 0
      NPrmAO = 0
      NShlDB = 0
      NPrmDB = 0
      NBTot = 0
      Call IClear(NAtoms,IAn)
      If(NLab.ge.3) Read(IU) IAn
      Call IClear(NAtoms,IAtTyp)
      If(NLab.ge.4) Read(IU) IAtTyp
      Call AClear(NAtoms,AtmChg)
      If(NLab.ge.5) Read(IU) AtmChg
      Call AClear(3*NAtoms,C)
      If(NLab.ge.6) Read(IU) (C(I),I=1,3*NAtoms)
      Call IClear(NBasis,IBfAtm)
      Call IClear(NBasis,IBfTyp)
      If(NLab.ge.7) Read(IU) IBfAtm,IBfTyp
        Do 10 I = 1, NAtoms
   10     AtmWgt(I) = 0.0d0
      If(NLab.ge.8) then
        Read(IU) AtmWgt
        If(NLab.ge.9) then
          Read(IU) NFC,NFV,ITran,IDum9
          If(NLab.ge.11) then
            Read(IU)
            Read(IU) NShlAO,NPrmAO,NShlDB,NPrmDB,NBTot
            Do 20 I = 12, NLab
              Read(IU)
   20         Continue
            endIf
          endIf
        endIf
      Return
      End
*Deck Rd_HeadA
      Subroutine Rd_HeadA(IU,NLab,NAtoms,NBasis,IAn,IAtTyp,AtmChg,C,
     $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,Rec11)
      Implicit None
      Integer LRec11
      Parameter (LRec11=16)
      Integer IU,NLab,NAtoms,NBasis,IAn(NAtoms),IAtTyp(NAtoms),
     $  IBfAtm(NBasis),IBfTyp(NBasis),NFC,NFV,ITran,IDum9,I,
     $  Rec11(LRec11)
      Real*8 AtmChg(NAtoms),C(3*NAtoms),AtmWgt(NAtoms)
CF2PY Intent(Out) IAn,IAtTyp,AtmChg,C,IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran
CF2PY Intent(Out) IDum9,Rec11
C
      NFC = 0
      NFV = 0
      ITran = 0
      IDum9 = 0
      Call IClear(LRec11,Rec11)
      Call IClear(NAtoms,IAn)
      If(NLab.ge.3) Read(IU) IAn
      Call IClear(NAtoms,IAtTyp)
      If(NLab.ge.4) Read(IU) IAtTyp
      Call AClear(NAtoms,AtmChg)
      If(NLab.ge.5) Read(IU) AtmChg
      Call AClear(3*NAtoms,C)
      If(NLab.ge.6) Read(IU) (C(I),I=1,3*NAtoms)
      Call IClear(NBasis,IBfAtm)
      Call IClear(NBasis,IBfTyp)
      If(NLab.ge.7) Read(IU) IBfAtm,IBfTyp
        Do 10 I = 1, NAtoms
   10     AtmWgt(I) = 0.0d0
      If(NLab.ge.8) then
        Read(IU) AtmWgt
        If(NLab.ge.9) then
          Read(IU) NFC,NFV,ITran,IDum9
          If(NLab.ge.11) then
            Read(IU)
            Read(IU) Rec11
            Do 20 I = 12, NLab
              Read(IU)
   20         Continue
            endIf
          endIf
        endIf
      Return
      End
*Deck Rd_IBuf
      Subroutine Rd_IBuf(IU,LR,LenBuf,Arr)
      Implicit None
      Integer IU,LR,LenBuf,Buf(LenBuf),Arr(LR),Ind,NDo,I
CF2PY Intent (Out) Arr
C
      Do 20 Ind = 0, (LR-1), LenBuf
        NDo = Min(LenBuf,LR-Ind)
        Read(IU) Buf
        Do 10 I = 1, NDo
   10     Arr(Ind+I) = Buf(I)
   20   Continue
      Return
      End
*Deck Rd_Labl
      Subroutine Rd_Labl(IU,IVers,CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,
     $  TypeA,NRI,EOF)
      Implicit None
      Integer IVers,LStr,IU,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,TypeA,NRI
      Parameter (LStr=64)
C     The latest f2py is now "improved" and is now too stupid to handle
C     character string lengths given by parameters.
C     Character*(LStr) CBuf
      Character*64 CBuf
      Logical EOF
CF2PY Intent (Out) NI,NR,NRI,NTot,LenBuf,N1,N2,N3,N4,N5,TypeA,EOF
CF2PY Intent (Out) CBuf
C
      CBuf = ' '
      NI = 0
      NR = 0
      NTot = 0
      LenBuf = 0
      N1 = 0
      N2 = 0
      N3 = 0
      N4 = 0
      N5 = 0
      TypeA = 0
      If(IVers.eq.1) then
        Read(IU,End=900) CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5
      else
        Read(IU,End=900) CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,TypeA
        endIf
      Call Clean_Label(CBuf)
      EOF = CBuf.eq.'END'
      If(NR.ge.0) then
        NRI = 1
      else
        NRI = 2
        NR = -NR
        endIf
      If(.not.EOF) then
        If(N2.eq.0) N2 = 1
        If(N3.eq.0) N3 = 1
        If(N4.eq.0) N4 = 1
        If(N5.eq.0) N5 = 1
        endIf
      Return
  900 CBuf = 'END'
      EOF = .True.
      NI = 0
      NR = 0
      NRI = 1
      NTot = 0
      LenBuf = 0
      N1 = 0
      N2 = 0
      N3 = 0
      N4 = 0
      N5 = 0
      TypeA = 0
      Return
      End
*Deck Rd_RBuf
      Subroutine Rd_RBuf(IU,LR,LenBuf,Arr)
      Implicit None
      Integer IU,LR,LenBuf,Ind,NDo,I
      Real*8 Buf(LenBuf),Arr(LR)
CF2PY Intent (Out) Arr
C
      Do 20 Ind = 0, (LR-1), LenBuf
        NDo = Min(LenBuf,LR-Ind)
        Read(IU) Buf
        Do 10 I = 1, NDo
   10     Arr(Ind+I) = Buf(I)
   20   Continue
      Return
      End
*Deck Rd_RInd
      Subroutine Rd_RInd(IU,NR,LR,NRLR,NTot,LenBuf,RArr)
      Implicit None
      Integer IU,NR,LR,NRLR,NTot,LenBuf,I,J,Ind,NDo,IO,IBuf(LenBuf)
      Real*8 Buf(NR,LenBuf),RArr(NRLR),Zero
      Parameter (Zero=0.0d0)
CF2PY Intent (Out) RArr
C
      Do 10 I = 1, LR*NR
   10   RArr(I) = Zero
      Do 50 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        If(NDo.eq.LenBuf.or..True.) then
          Read(IU) IBuf, Buf
        else
C         For debugging
          Read(IU) (IBuf(I),I=1,NDo), ((Buf(J,I),J=1,NR),I=1,NDo)
          endIf
        Do 40 I = 1, NDo
          IO = IBuf(I)
          Do 30 J = 1, NR
   30       RArr(J+(IO-1)*NR) = Buf(J,I)
   40     Continue
   50   Continue
      Return
      End
*Deck Rd_Skip
      Subroutine Rd_Skip(IU,NTot,LenBuf)
      Implicit None
      Integer IU,NTot,LenBuf,I,NRec
C
      If(NTot.eq.0) Return
      NRec = (NTot+LenBuf-1)/LenBuf
      Do 10 I = 1, NRec
        Read(IU)
   10   Continue
      Return
      End
*Deck Wr_2E
      Subroutine Wr_2E(IU,NTot,NR,N,LR,LenBuf,RArr)
      Implicit None
      Logical NonZ, SqArr
      Integer IU,NR,N,LR,LenBuf,IBuf(4,LenBuf),I,J,K,L,LimL,IJKL,
     $  IB,IR,NNZ,NTot,NTT,NT4,IS,IJS,IJKS
      Real*8 RArr(LR,NR),RBuf(NR,LenBuf),Zero
      Parameter (Zero=0.0d0)
C
      NTT = (N*(N+1))/2
      NT4 = (NTT*(NTT+1))/2
      SqArr = LR.eq.(N*N*N*N)
      If(LR.ne.NT4.and..not.SqArr)
     $  Stop 'Unrecognized array dimension in Wr_2E'
      NNZ = 0
      IB = 0
      IJKL = 0
      Do 60 I = 1, N
        IS = (I-1)*N
        Do 50 J = 1, I
          IJS = (IS+J-1)*N
          Do 40 K = 1, I
            If(I.eq.K) then
              LimL = J
            else
              LimL = K
              endIf
            IJKS = (IJS+K-1)*N
            Do 30 L = 1, LimL
              If(SqArr) then
                IJKL = L + IJKS
              else
                IJKL = IJKL + 1
                endIf
              NonZ = RArr(IJKL,1).ne.Zero
              Do 10 IR = 2, NR
   10           NonZ = NonZ.or.RArr(IJKL,IR).ne.Zero
              If(NonZ) then
                IB = IB + 1
                IBuf(1,IB) = I
                IBuf(2,IB) = J
                IBuf(3,IB) = K
                IBuf(4,IB) = L
                Do 20 IR = 1, NR
   20             RBuf(IR,IB) = RArr(IJKL,IR)
                If(IB.eq.LenBuf) then
                  Write(IU) IBuf, RBuf
                  NNZ = NNZ + LenBuf
                  IB = 0
                  endIf
                endIf
   30         Continue
   40       Continue
   50     Continue
   60   Continue
      If(IB.gt.0.or.NNZ.eq.0) then
C       IB should only be = 0 if no non-zero elements were found
        If(IB.eq.0) then
C     If no data is written, write the first value as index 1 w/ zero values
          IBuf(1,1) = 1
          IBuf(2,1) = 1
          IBuf(3,1) = 1
          IBuf(4,1) = 1
          IB = 1
          Do 70 IR = 1, NR
   70       RBuf(IR,1) = Zero
          Endif
C       Zero out the rest of the buffers
        Do 90 I = (IB+1), LenBuf
          IBuf(1,I) = 0
          IBuf(2,I) = 0
          IBuf(3,I) = 0
          IBuf(4,I) = 0
          Do 80 IR = 1, NR
   80       RBuf(IR,I) = Zero
   90     Continue
        Write(IU) IBuf, RBuf
        NNZ = NNZ + IB
        endIf
      If(NNZ.ne.NTot.and.NTot.ne.-1) Stop 'NNZ not NTot in Wr_2E'
      Return
      End
*Deck Wr_2EP
      Subroutine Wr_2EP(IU,NTot,NR,N,LR,LenBuf,LRNR,RArr)
      Implicit None
      Integer IU,NTot,NR,N,LR,LenBuf,LRNR
      Real*8 RArr(LRNR)
C
      Call Wr_2E(IU,NTot,NR,N,LR,LenBuf,RArr)
      Return
      End
*Deck Wr_SpA
      Subroutine Wr_SpA(IU,NI,NR,NTot,LenBuf,IArr,RArr)
      Implicit None
      Integer IU,NI,NR,NTot,LenBuf,IBuf(NI,LenBuf),I,J,
     $  IArr(NI*NTot),IndI,IndR,Ind,NDo
      Real*8 RArr(NR*NTot),Buf(NR,LenBuf)
C
      Do 70 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        IndI = Ind*NI
        IndR = Ind*NR
        Do 60 I = 1, NDo
          Do 40 J = 1, NI
   40       IBuf(J,I) = IArr(J+IndI)
          IndI = IndI + NI
          Do 50 J = 1, NR
   50       Buf(J,I) = RArr(J+IndR)
   60     IndR = IndR + NR
        Call IClear((LenBuf-NDo)*NI,IBuf(1,NDo+1))
        Call AClear((LenBuf-NDo)*NR,Buf(1,NDo+1))
        Write(IU) IBuf, Buf
   70   Continue
      Return
      End
*Deck Wr_ChBuf
      Subroutine Wr_ChBuf(IU,LR,LenBuf,CArr)
      Implicit None
      Integer IU,LR,LenBuf,Ind,NDo,Lim
      Character CArr*(*), Buf*(LenBuf)
C
      Do 20 Ind = 0, (LR-1), LenBuf
        NDo = Min(LenBuf,LR-Ind)
        Lim = Min(Ind+NDo,LR)
        Buf = CArr(Ind+1:Lim)
        Write(IU) Buf
   20   Continue
      Return
      End
*Deck Wr_CBuf
      Subroutine Wr_CBuf(IU,LR,LenBuf,Arr)
      Implicit None
      Integer IU,LR,LenBuf,Ind,NDo,I
      Complex*16 Buf(LenBuf),Arr(LR),Zero
C     The latest f2py is now "improved" and is now too stupid to handle
C     complex variables in parameter statements.
C     Parameter (Zero=(0.0d0,0.0d0))
      Real*8 ZeroR
      Parameter (ZeroR=0.0d0)
C
      Zero = DCmplx(ZeroR,ZeroR)
      Do 30 Ind = 0, (LR-1), LenBuf
        NDo = Min(LenBuf,LR-Ind)
        Do 10 I = 1, NDo
   10     Buf(I) = Arr(Ind+I)
        Do 20 I = (NDo+1), LenBuf
   20     Buf(I) = Zero
        Write(IU) Buf
   30   Continue
      Return
      End
*Deck Wr_Head
      Subroutine Wr_Head(IU,NAtoms,NAt3,NBasis,IAn,IAtTyp,AtmChg,C,
     $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,NShlDB,
     $  NPrmDB,NBTot)
      Implicit None
      Integer LRec11,IZero
      Parameter (LRec11=16,IZero=0)
      Integer IU,NAtoms,NAt3,NBasis,IAn(NAtoms),IAtTyp(NAtoms),
     $  IBfAtm(NBasis),IBfTyp(NBasis),NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,
     $  NShlDB,NPrmDB,NBTot,I,Rec11(LRec11),LRec(2)
      Real*8 AtmChg(NAtoms),C(NAt3),AtmWgt(NAtoms)
C
      If(Mod(NAtoms,2).eq.1) then
        Write(IU) IAn,IZero
        Write(IU) IAtTyp,IZero
      else
        Write(IU) IAn
        Write(IU) IAtTyp
        endIf
      Write(IU) AtmChg
      Write(IU) (C(I),I=1,3*NAtoms)
      Write(IU) IBfAtm,IBfTyp
      Write(IU) AtmWgt
      Write(IU) NFC,NFV,ITran,IDum9
      LRec(1) = LRec11
      LRec(2) = 0
      Write(IU) LRec
      Do 10 I = 1, LRec11
   10   Rec11(I) = 0
      Rec11(1) = NShlAO
      Rec11(2) = NPrmAO
      Rec11(3) = NShlDB
      Rec11(4) = NPrmDB
      Rec11(5) = NBTot
      Write(IU) Rec11
      Return
      End
*Deck Wr_HeadA
      Subroutine Wr_HeadA(IU,NAtoms,NAt3,NBasis,IAn,IAtTyp,AtmChg,C,
     $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,Rec11)
      Implicit None
      Integer LRec11,IZero
      Parameter (LRec11=16,IZero=0)
      Integer IU,NAtoms,NAt3,NBasis,IAn(NAtoms),IAtTyp(NAtoms),
     $  IBfAtm(NBasis),IBfTyp(NBasis),NFC,NFV,ITran,IDum9,I,
     $  Rec11(LRec11),LRec(2)
      Real*8 AtmChg(NAtoms),C(NAt3),AtmWgt(NAtoms)
C
      If(Mod(NAtoms,2).eq.1) then
        Write(IU) IAn,IZero
        Write(IU) IAtTyp,IZero
      else
        Write(IU) IAn
        Write(IU) IAtTyp
        endIf
      Write(IU) AtmChg
      Write(IU) (C(I),I=1,3*NAtoms)
      Write(IU) IBfAtm,IBfTyp
      Write(IU) AtmWgt
      Write(IU) NFC,NFV,ITran,IDum9
      LRec(1) = LRec11
      LRec(2) = 0
      Write(IU) LRec
      Write(IU) Rec11
      Return
      End
*Deck Wr_DummyHead
      Subroutine Wr_DummyHead(IU)
      Implicit None
      Integer IU,LRec9,LRec11,IZero,NAtoms,NBasis,I
      Parameter (LRec9=4,LRec11=16,IZero=0,NAtoms=1,NBasis=1)
      Integer IAn(NAtoms),IAtTyp(NAtoms),IBfAtm(NBasis),
     $  IBfTyp(NBasis),Rec9(LRec9),Rec11(LRec11),LRec(2)
      Real*8 AtmChg(NAtoms),C(3,NAtoms),AtmWgt(NAtoms)
C
      IAn(1) = 2
      IAtTyp(1) = 0
      AtmChg(1) = 2.0d0
      C(1,1) = 0.0d0
      C(2,1) = 0.0d0
      C(3,1) = 0.0d0
      IBfAtm(1) = 1
      IBfTyp(1) = 1
      AtmWgt(1) = 4.0d0
      Do 10 I = 1, LRec9
   10   Rec9(I) = 0
      Do 20 I = 1, LRec11
   20   Rec11(I) = 0
      Write(IU) IAn,IZero
      Write(IU) IAtTyp,IZero
      Write(IU) AtmChg
      Write(IU) C
      Write(IU) IBfAtm,IBfTyp
      Write(IU) AtmWgt
      Write(IU) Rec9
      LRec(1) = LRec11
      LRec(2) = 0
      Write(IU) LRec
      Write(IU) Rec11
      Return
      End
*Deck Wr_IBuf
      Subroutine Wr_IBuf(IU,LR,LenBuf,Arr)
      Implicit None
      Integer IU,LR,LenBuf,Buf(LenBuf),Arr(LR),Ind,NDo,I
C
      Do 30 Ind = 0, (LR-1), LenBuf
        NDo = Min(LenBuf,LR-Ind)
        Do 10 I = 1, NDo
   10     Buf(I) = Arr(Ind+I)
        Do 20 I = (NDo+1), LenBuf
   20     Buf(I) = 0
        Write(IU) Buf
   30   Continue
      Return
      End
*Deck Wr_Labl
      Subroutine Wr_Labl(IU,CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,TypeA)
      Implicit None
      Integer LStr,IU,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,TypeA
      Parameter (LStr=64)
C     The latest f2py is now "improved" and is now too stupid to handle
C     character string lengths given by parameters.
C     Character CBuf*(*), CBufL*(LStr)
      Character CBuf*(*), CBufL*64
C
      CBufL = CBuf
      Call Clean_Label(CBufL)
      Write(IU) CBufL,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,TypeA
      Return
      End
*Deck Wr_RBuf
      Subroutine Wr_RBuf(IU,LR,LenBuf,Arr)
      Implicit None
      Integer IU,LR,LenBuf,Ind,NDo,I
      Real*8 Buf(LenBuf),Arr(LR),Zero
      Parameter (Zero=0.0d0)
C
      Do 30 Ind = 0, (LR-1), LenBuf
        NDo = Min(LenBuf,LR-Ind)
        Do 10 I = 1, NDo
   10     Buf(I) = Arr(Ind+I)
        Do 20 I = (NDo+1), LenBuf
   20     Buf(I) = Zero
        Write(IU) Buf
   30   Continue
      Return
      End
*Deck Wr_RInd
      Subroutine Wr_RInd(IU,NR,LR,NTot,LenBuf,RArr)
      Implicit None
      Logical NonZer
      Integer IU,NR,LR,NTot,LenBuf,I,J,IBuf(LenBuf),IB,NNZ
      Real*8 Buf(NR,LenBuf),RArr(NR,LR),Zero
      Parameter (Zero=0.0d0)
C
      NNZ = 0
      IB = 0
      Do 40 I = 1, LR
        NonZer = .False.
        Do 30 J = 1, NR
          Buf(J,IB+1) = RArr(J,I)
   30     NonZer = NonZer.or.Buf(J,IB+1).ne.Zero
        If(NonZer) then
          IB = IB + 1
          IBuf(IB) = I
          If(IB.eq.LenBuf) then
            Write(IU) IBuf, Buf
            IB = 0
            NNZ = NNZ + LenBuf
            endIf
          endIf
   40   Continue
C     If no data is written, write the first value as index 1 w/ zero values
      If(IB.gt.0.or.NNZ.eq.0) then
C       IB should only be = 0 if no non-zero elements were found
        If(IB.eq.0) then
          IBuf(1) = 1
          Do 50 J = 1, NR
   50       Buf(J,1) = Zero
          IB = 1
          Endif
        NNZ = NNZ + IB
C       Zero the rest of the buffers
        Do 70 I = (IB+1), LenBuf
          IBuf(I) = 0
          Do 60 J = 1, NR
   60       Buf(J,I) = Zero
   70     Continue
        Write(IU) IBuf, Buf
        endIf
      If(NNZ.ne.NTot) Stop 'NZ error in Wr_RInd'
      Return
      End
*Deck Wr_RIndP
      Subroutine Wr_RIndP(IU,NR,LR,NTot,LenBuf,RArr)
      Implicit None
      Integer IU,NR,LR,NTot,LenBuf
      Real*8 RArr(NR*LR)

      Call Wr_RInd(IU,NR,LR,NTot,LenBuf,RArr)

      Return
      End
*Deck ExpAO1
      Subroutine ExpAO1(N,LR,RI,RO)
      Implicit None
      Integer N,LR,I,J,K,L,LimL,IJKL
      Real*8 RI(LR), RO(N,N,N,N), R
CF2PY Intent (Out) RO
C
      IJKL = 0
      Do 40 I = 1, N
        Do 30 J = 1, I
          Do 20 K = 1, I
            If(I.eq.K) then
              LimL = J
            else
              LimL = K
              endIf
            Do 10 L = 1, LimL
              R = RI(IJKL+L)
              RO(I,J,K,L) = R
              RO(I,J,L,K) = R
              RO(J,I,K,L) = R
              RO(J,I,L,K) = R
              RO(K,L,I,J) = R
              RO(L,K,I,J) = R
              RO(K,L,J,I) = R
   10         RO(L,K,J,I) = R
   20       IJKL = IJKL + LimL
   30     Continue
   40   Continue
      Return
      End
*Deck ExpAON
      Subroutine ExpAON(NE,N,LR,RI,RO)
      Implicit None
      Integer N,LR,I,J,K,L,LimL,IJKL,IE,NE
      Real*8 RI(NE,LR), RO(NE,N,N,N,N), R
CF2PY Intent (Out) RO
C
      IJKL = 0
      Do 40 I = 1, N
        Do 30 J = 1, I
          Do 20 K = 1, I
            If(I.eq.K) then
              LimL = J
            else
              LimL = K
              endIf
            Do 10 L = 1, LimL
              Do 5 IE = 1, NE
                R = RI(IE,IJKL+L)
                RO(IE,I,J,K,L) = R
                RO(IE,I,J,L,K) = R
                RO(IE,J,I,K,L) = R
                RO(IE,J,I,L,K) = R
                RO(IE,K,L,I,J) = R
                RO(IE,L,K,I,J) = R
                RO(IE,K,L,J,I) = R
    5           RO(IE,L,K,J,I) = R
   10         Continue
   20       IJKL = IJKL + LimL
   30     Continue
   40   Continue
      Return
      End
*Deck QCM_ExpAON
      Subroutine QCM_ExpAON(NE,N,LR,NELR,NOut,RI,RO)
      Implicit None
      Integer NE, N, LR, NELR, NOut
      Real*8 RI(NELR), RO(NOut)
CF2PY Intent (Out) RO
C
      Call ExpAON(NE,N,LR,RI,RO)
      Return
      End
*Deck AClear
      Subroutine AClear(N,A)
      Implicit None
C
C     Routine to clear N elements in array A.
C
      Integer N, I
      Real*8 Zero, A(N)
      Parameter (Zero=0.0D0)
C
      Do 10 I = 1, N
   10   A(I) = Zero
      Return
      End
*Deck IClear
      Subroutine IClear(N,IA)
      Implicit None
C
C     Routine to clear N elements in array IA.
C
      Integer N,IA(N),I
C
      Do 10 I = 1, N
   10   IA(I) = 0
      Return
      End
*Deck Clean_Label
      Subroutine Clean_Label(CBuf)
      Implicit None
C
C     Canonicalize the label in CBuf by removing leading and trailing
C     spaces, collapsing multiple spaces, and captializing.
C
      Character CBuf*(*), Char*1, IC
      Logical InBl
      Integer IChar, ICI, IUA, ILA, ILZ, Delta, LB, L, I
C
      IUA = IChar('A')
      ILA = IChar('a')
      ILZ = IChar('z')
      Delta = IUA - ILA
      InBl = .True.
      L = 0
      LB = Len(CBuf)
      Do 10 I = 1, LB
        IC = CBuf(I:I)
        If(IC.eq.' ') then
          If(.not.InBl) then
            InBl = .True.
            L = L + 1
            CBuf(L:L) = ' '
            endIf
        else
          InBl = .False.
          ICI = IChar(IC)
          If(ICI.ge.ILA.and.ICI.le.ILZ) ICI = ICI + Delta
          L = L + 1
          CBuf(L:L) = Char(ICI)
          endIf
   10   Continue
      If(L.lt.LB) CBuf(L+1:) = ' '
      Return
      End
