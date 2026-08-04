*Deck Wr_LIBuf
      Subroutine Wr_LIBuf(IU,Label,NI,LenBuf,N1,N2,N3,N4,N5,TypeA,IX)
      Implicit None
C
C     This file provides some useful higher-level routines for Fortran
C     users of qcmatrixio.F.  These are in a separate file because for
C     these operations it is better to have separate Fortran and Python
C     wrappers since variable-length arrays make it cumbersome to
C     wrap the Fortran versions and call them from Python.
C
C     The Fortran routines which are used both here and in the
C     Python interface are in qcmatrixio.F.
C
C     Routines provided here:
C     Call Wr_LIBuf(IU,Label,NI,LenBuf,N1,N2,N3,N4,N5,TypeA,X)
C     Call Wr_LRBuf(IU,Label,NI,LenBuf,N1,N2,N3,N4,N5,TypeA,X)
C     Call Wr_LCBuf(IU,Label,NI,LenBuf,N1,N2,N3,N4,N5,TypeA,X)
C     Call Wr_LChBuf(IU,Label,LenBuf,X)
C     Write integer/real*8/complex array or character string X unblocked
C     (zeros or blanks included) to unit IU given the parameters from the
C     header record.
C
C     Call Wr_LRInd(IU,Label,NR,LenBuf,N1,N2,N3,N4,N5,TypeA,X)
C     Write a real array compressed, with indices for non-zero
C     sets of (NR) elements.
C
      Character*(*) Label
      Integer IU,NI,LenBuf,N1,N2,N3,N4,N5,IX(*),LenBX,LenArr,NTot,TypeA
C
      NTot = LenArr(N1,N2,N3,N4,N5)
      LenBX = (LenBuf/NI)*NI
      Call Wr_Labl(IU,Label,NI,0,NTot,LenBX,N1,N2,N3,N4,N5,TypeA)
      Call Wr_IBuf(IU,NTot*NI,LenBX,IX)
      Return
      End
*Deck Wr_LRBuf
      Subroutine Wr_LRBuf(IU,Label,NR,LenBuf,N1,N2,N3,N4,N5,TypeA,X)
      Implicit None
      Character*(*) Label
      Integer IU,NR,LenBuf,N1,N2,N3,N4,N5,LenBX,LenArr,NTot,TypeA
      Real*8 X(*)
C
      NTot = LenArr(N1,N2,N3,N4,N5)
      LenBX = (LenBuf/NR)*NR
      Call Wr_Labl(IU,Label,0,NR,NTot,LenBX,N1,N2,N3,N4,N5,TypeA)
      Call Wr_RBuf(IU,NTot*NR,LenBX,X)
      Return
      End
*Deck Wr_LCBuf
      Subroutine Wr_LCBuf(IU,Label,NR,LenBuf,N1,N2,N3,N4,N5,TypeA,X)
      Implicit None
      Character*(*) Label
      Integer IU,NR,LenBuf,N1,N2,N3,N4,N5,LenBX,LenArr,NTot,TypeA
      Complex*16 X(*)
C
      NTot = LenArr(N1,N2,N3,N4,N5)
      LenBX = (LenBuf/(2*NR))*NR
      Call Wr_Labl(IU,Label,0,-NR,NTot,LenBX,N1,N2,N3,N4,N5,TypeA)
      Call Wr_CBuf(IU,NTot*NR,LenBX,X)
      Return
      End
*Deck Wr_LChBuf
      Subroutine Wr_LChBuf(IU,Label,LenBuf,X)
      Implicit None
      Character*(*) Label,X
      Integer IU,LenBuf,LenBX,NTot,Len4L,LR
#ifdef USE_I8
      Parameter (Len4L=8)
#else
      Parameter (Len4L=4)
#endif
C
      LR = Len(X)
      NTot = (LR+Len4L-1)/Len4L
      LenBX = LenBuf*Len4L
      Call Wr_Labl(IU,Label,1,0,NTot,LenBuf,NTot,0,0,0,0,1)
      Call Wr_ChBuf(IU,LR,LenBX,X)
      Return
      End
*Deck Wr_LRInd
      Subroutine Wr_LRInd(IU,Label,NR,LenBuf,N1,N2,N3,N4,N5,TypeA,X)
      Implicit None
      Character*(*) Label
      Integer IU,NR,LenBuf,N1,N2,N3,N4,N5,LenBX,NTot,LenArr,LR,
     $  QCM_NumNZR,TypeA
      Real*8 X(NR,*)
C
      LR = LenArr(N1,N2,N3,N4,N5)
      LenBX = LenBuf/NR
      NTot = QCM_NumNZR(NR,LR,X)
      Call Wr_Labl(IU,Label,1,NR,NTot,LenBX,N1,N2,N3,N4,N5,TypeA)
      Call Wr_RInd(IU,NR,LR,NTot,LenBX,X)
      Return
      End
*Deck Wr_LAO2E
      Subroutine Wr_LAO2E(IU,Label,NR,LenBuf,N,RInt)
      Implicit None
      Character*(*) Label
      Integer IU,NR,LenBuf,N,LR,LenArr,LenBX,NTot,QCM_NumNZT,I0
      Real*8 RInt(*)
      Parameter (I0=0)
C
      LR = LenArr(-N,-N,-N,N,1)
      LenBX = LenBuf/(NR+2)
      NTot = QCM_NumNZT(NR,LR,RInt)
      Call Wr_Labl(IU,Label,4,NR,NTot,LenBX,-N,-N,-N,N,1,I0)
      Call Wr_2E(IU,NTot,NR,N,LR,LenBX,RInt)
      Return
      End
*Deck ExpD2E
      Subroutine ExpD2E(NAt,N,NTot,RD2E,ID2E,D2E)
      Implicit None
      Integer NAt,N,NTot,IA,I,J,K,L,ID,ID2E(5,NTot)
      Real*8 RD2E(3,NTot), D2E(N,N,N,N,3,NAt), R1, R2, R3
C
      Call AClear(3*NAt*N**4,D2E)
      Do 10 ID = 1, NTot
        I = ID2E(1,ID)
        J = ID2E(2,ID)
        K = ID2E(3,ID)
        L = ID2E(4,ID)
        IA = ID2E(5,ID)
        R1 = RD2E(1,ID)
        R2 = RD2E(2,ID)
        R3 = RD2E(3,ID)
        D2E(L,K,J,I,1,IA) = R1
        D2E(K,L,J,I,1,IA) = R1
        D2E(L,K,I,J,1,IA) = R1
        D2E(K,L,I,J,1,IA) = R1
        D2E(J,I,L,K,1,IA) = R1
        D2E(J,I,K,L,1,IA) = R1
        D2E(I,J,L,K,1,IA) = R1
        D2E(I,J,K,L,1,IA) = R1
        D2E(L,K,J,I,2,IA) = R2
        D2E(K,L,J,I,2,IA) = R2
        D2E(L,K,I,J,2,IA) = R2
        D2E(K,L,I,J,2,IA) = R2
        D2E(J,I,L,K,2,IA) = R2
        D2E(J,I,K,L,2,IA) = R2
        D2E(I,J,L,K,2,IA) = R2
        D2E(I,J,K,L,2,IA) = R2
        D2E(L,K,J,I,3,IA) = R3
        D2E(K,L,J,I,3,IA) = R3
        D2E(L,K,I,J,3,IA) = R3
        D2E(K,L,I,J,3,IA) = R3
        D2E(J,I,L,K,3,IA) = R3
        D2E(J,I,K,L,3,IA) = R3
        D2E(I,J,L,K,3,IA) = R3
   10   D2E(I,J,K,L,3,IA) = R3
      Return
      End
*Deck Pr_SpA
      Subroutine Pr_SpA(IOut,NI,NRI,NR,NTot,IArr,RArr)
      Implicit None
C
C     Print a pair of matrices with arbitrary dimensions.
C
      Integer IOut, NI, NRI, NR, NTot, IArr(NI,NTot), I, J, IRI, NRT
      Real*8 RArr(NRI,NR,NTot)
      Character*80 Fmt
 1000 Format('Pr_SpA:  NI=',I8,' NRI=',I8,' NR=',I8,' too large.')
 1010 Format('(1X,',I2,'I9,',I2,'F12.6)')
C
      NRT = NRI*NR
      If(NI.gt.99.or.NRT.gt.99) then
        Write(IOut,1000) NI, NRI, NR
      else
        Fmt = ' '
        Write(Fmt,1010) NI, NRT
        Do 10 I = 1, NTot
          Write(IOut,Fmt) (IArr(J,I),J=1,NI),
     $      ((RArr(IRI,J,I),IRI=1,NRI),J=1,NR)
   10     Continue
        endIf
      Return
      End
