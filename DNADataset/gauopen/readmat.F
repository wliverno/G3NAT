*Deck ReadFaf
      Program ReadFaf
      Implicit Real*8(A-H,O-Z)
C
C     This example program reads a binary array file.  It illustrates
C     use of the Fortran interface.  The name of the file is the first
C     argument on the command line.
C
      Parameter (LStr=64,IOut=6,MaxAt=30000,MaxBf=30000,MaxArr=1000000000,
     $  MaxAr5=(MaxArr/5),MaxChArr=10000000,NCPLine=80,LRec11=16)
      Character*(LStr) LabFil, GVers, Title, CBuf, FileName*80,
     $  ChArr*(MaxChArr)
      Logical EOF, AOInts, DAOInts
      Integer QCM_MaxINZR,IRec11(LRec11)
      Dimension IAn(MaxAt),IAtTyp(MaxAt),AtmChg(MaxAt),C(3,MaxAt),
     $  IBfAtm(MaxBf),IBfTyp(MaxBf),AtmWgt(MaxAt),IArr(MaxArr),
     $  RArr(MaxArr),RArr3(3,MaxAr5),IArr5(5,MaxAr5)
      Complex*16 CArr(MaxArr)
      Equivalence (IArr(1),IArr5(1,1)),(RArr(1),CArr(1),RArr3(1,1))
 1000 Format(' File ',A,' IU=',I6)
 1010 Format(' Label ',A,' IVers=',I2,' NLab=',I2,' Version=',A,/,
     $  ' Title ',A,/,' NAtoms=',I6,' NBasis=',I6,' NBsUse=',I6,
     $  ' ICharg=',I6,' Multip=',I6,' NE=',I6,' Len12L=',I1,' Len4L=',
     $  I1,' IOpCl=',I6,' ICGU=',I3)
 1030 Format('    IAn=',20I4)
 1040 Format(' IAtTyp=',5I12)
 1050 Format(' AtmChg=',5F12.6)
 1060 Format('      C=',3F12.6)
 1070 Format(' IBfAtm=',10I8)
 1080 Format(' IBfTyp=',10I8)
 1090 Format(' AtmWgt=',5F12.6)
 1100 Format(' NFC=',I6,' NFV=',I6,' ITran=',I6,' IDum9=',I6,/,
     $  ' NShlAO=',I7,' NPrmAO=',I7,' NShlDB=',I7,' NPrmDB=',I7,
     $  '  NBTot=',I7,' NDeriv=',I7,'   ND12=',I7,' NFrqPt=',I7,/,
     $  ' NShNEO=',I7,' NPrNEO=',I7,' R11-11=',I7,' R11-12=',I7,
     $  ' R11-13=',I7,' R11-14=',I7,' R11-15=',I7,' R11-16=',I7)
 1110 Format(1x,A40,' NI=',I2,' NR=',I2,' NRI=',I2,' NTot=',I8,
     $  ' LenBuf=',I5,' N=',5I6,' IType=',I5)
 1120 Format(' Length=',I12,' exceeds MaxArr=',I12,'.')
 1140 Format(' LR=',I12,' but NTot=',I12,'.')
 1150 Format(1X,10I12)
 1160 Format(1X,10F12.6)
 1170 Format(1X,10F12.6)
 1180 Format(5I9,3D15.6)
 1190 Format(1X,A)
 1200 Format(' NAtoms=',I7,' NBasis=',I8,' but MaxAt=',I7,' MaxBf=',I8,
     $  '.')
 1210 Format(' D2E NTot=',I10,' exceeds dimension of',I10,' skipped,')
C
      FileName = ' '
      Call Get_Command_Argument(1,FileName,LFN,IStat)
      If(IStat.ne.0) then
        Write(6,*) 'Error fetching argument, status=',IStat
        Stop
        endIf
C
C     Open the file.  IU is returned with the Fortran unit number
C     on which the file is open.
C
      Call Open_Read(FileName(1:LFN),IU,LabFil,IVers,NLab,GVers,Title,
     $  NAtoms,NBasis,NBsUse,ICharg,Multip,NE,Len12L,Len4L,IOpCl,ICGU)
      Write(IOut,1000) FileName(1:LFN), IU
      If(IU.le.0) Stop
      LLab = Len_Trim(LabFil)
      LVers = Len_Trim(GVers)
      LTit = Len_Trim(Title)
      Write(IOut,1010) LabFil(1:LLab),IVers,NLab,GVers(1:LVers),
     $  Title(1:LTit),NAtoms,NBasis,NBsUse,ICharg,Multip,NE,Len12L,
     $  Len4L,IOpCl,ICGU
      If(NAtoms.gt.MaxAt.or.NBasis.gt.MaxBF) then
        Write(IOut,1200) NAtoms, NBasis, MaxAt, MaxBf
        Stop
        endIf
C
C     Read the header records, which contain basic information about
C     the molecule such as number of atoms, atomic numbers, Cartesian
C     coordinates, and information about the basis functions.
C
      Call Rd_HeadA(IU,NLab,NAtoms,NBasis,IAn,IAtTyp,AtmChg,C,IBfAtm,
     $  IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,IRec11)
      Write(IOut,1030) (IAn(I),I=1,NAtoms)
      Write(IOut,1040) (IAtTyp(I),I=1,NAtoms)
      Write(IOut,1050) (AtmChg(I),I=1,NAtoms)
      Write(IOut,1060) ((C(J,I),J=1,3),I=1,NAtoms)
      Write(IOut,1070) (IBfAtm(I),I=1,NBasis)
      Write(IOut,1080) (IBfTyp(I),I=1,NBasis)
      Write(IOut,1090) (AtmWgt(I),I=1,NAtoms)
      Write(IOut,1100) NFC, NFV, ITran, IDum9, IRec11
C
C     Loop over objects in the file, printing each.  Any object we
C     do not recognize is skipped using Rd_Skip.
C
   10 Call Rd_Labl(IU,IVers,CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,IType,
     $  NRI,EOF)
      Write(IOut,1110) CBuf,NI,NR,NRI,NTot,LenBuf,N1,N2,N3,N4,N5,IType
      If(.not.EOF) then
        LR = LenArr(N1,N2,N3,N4,N5)
        If((NI*NR).eq.0.and.LR.ne.NTot) then
          Write(IOut,1140) LR, NTot
          Call Rd_Skip(IU,NTot,LenBuf)
        else if(LR.gt.MaxArr) then
          Write(IOut,1120) LR, MaxArr
          Call Rd_Skip(IU,NTot,LenBuf)
        else if(NI.eq.1.and.NR.eq.0) then
          If(IType.gt.0) then
            NTotC = Len4L*NTot
            LenBC = Len4L*LenBuf
            Call Rd_ChBuf(IU,NTotC,LenBC,ChArr)
            LPrt = Min(NTotC,Len(ChArr))
            Do 20 Ind = 1, LPrt, NCPLine
              Lim = Min(Ind+NCPLine-1,NTotC,Len(ChArr))
              Write(IOut,1190) ChArr(Ind:Lim)
   20         Continue
          else
            Call Rd_IBuf(IU,NTot,LenBuf,IArr)
            Write(IOut,1150) (IArr(I),I=1,LR)
            endIf
        else if(NI.eq.0.and.NR.eq.1.and.NRI.eq.1) then
          Call Rd_RBuf(IU,NTot,LenBuf,RArr)
          Write(IOut,1160) (RArr(I),I=1,LR)
        else if(NI.eq.0.and.NR.eq.1.and.NRI.eq.2) then
          Call Rd_CBuf(IU,NTot,LenBuf,CArr)
          Write(IOut,1170) (CArr(I),I=1,LR)
        else if(AOInts(CBuf)) then
          NLR = LR*NR
          If(NLR.le.MaxArr) then
            Call Rd_2EN(IU,NR,LR,NLR,NTot,LenBuf,RArr)
            Write(IOut,1160) (RArr(I),I=1,NLR)
          else
            Write(IOut,1210) NLR, MaxArr
            Call Rd_Skip(IU,NTot,LenBuf)
            endIf
        else if(DAOInts(CBuf)) then
          If(NTot.le.MaxAr5) then
            Call Rd_SpA(IU,NI,NR,NTot,LenBuf,IArr5,RArr3)
            Write(IOut,1180)
     $        ((IArr5(J,I),J=1,5),(RArr3(J,I),J=1,3),I=1,NTot)
          else
            Write(IOut,1210) NTot, MaxAr5
            Call Rd_Skip(IU,NTot,LenBuf)
            endIf
        else if(NI.eq.1) then
          If(NRI.eq.1) then
            Call Rd_RInd(IU,NR,LR,NR*LR,NTot,LenBuf,RArr)
            NMax = QCM_MaxINZR(NR,LR,RArr)
            Write(IOut,1160) (RArr(I),I=1,NR*NMax)
          else
            NMax = QCM_MaxINZR(NRI*NR,LR,RArr)
            Call Rd_RInd(IU,NRI*NR,LR,NRI*NR*LR,NTot,LenBuf,CArr)
            Write(IOut,1170) (CArr(I),I=1,NRI*NR*NMax)
            endIf
        else if(NI.gt.0.and.NR.gt.0) then
          Lim = Min(MaxArr/NI,MaxArr/(NR*NRI))
          If(NTot.le.Lim) then
            Call Rd_SpA(IU,NI,NRI*NR,NTot,LenBuf,IArr,RArr)
            Call Pr_SpA(IOut,NI,NRI,NR,NTot,IArr,RArr)
          else
            Write(IOut,1210) NTot, Lim
            Call Rd_Skip(IU,NTot,LenBuf)
            endIf
        else
          Call Rd_Skip(IU,NTot,LenBuf)
          endIf
        Goto 10
        endIf
      Call Close_FAF(.False.,IU)
      End
