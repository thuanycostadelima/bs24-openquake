c =======================================================================
c  bs24_runner.f
c
c  This is a program to run Bayless and Somerville (2024) Ground Motion Model for Australia (BS24).
c  Implemented by Thuany Costa de Lima ( thuany.costadelima@ga.gov.au), Geoscience Australia
c            
c           
c
c  This code contains:  
c              (1) S24_interp  — the interpolation subroutine that
c                                bs24_gmm.f calls internally
c                                bs24_gmm.f is the code that contains the subroutine
c              (2) BS24_runner — the main program that loops over
c                                a matrix of scenarios and writes
c                                results to a CSV file
c
c  Usage:
c    Compile:  gfortran -O2 -ffixed-line-length-none -o bs24_run
c                       bs24_runner.f bs24_gmm.f
c    Run:      ./bs24_run
c    Output:   bs24_fortran_output.csv
c
c  The -ffixed-line-length-none flag is required because bs24_gmm.f
c  contains lines longer than the standard 72-character Fortran limit.
c
c  Output columns:
c    CratFlag  — 1 = Cratonic, 0 = NonCratonic
c    mag       — moment magnitude
c    rrup      — closest distance to rupture (km)
c    vs30      — time-averaged shear-wave velocity (m/s)
c    ztor      — depth to top of rupture (km)
c    period    — spectral period (s); 0.0 = PGA
c    lnY_cmss  — ln(SA) in cm/s/s (Fortran native units)
c                 To convert to ln(g): subtract 6.89
c    sigma     — total standard deviation (ln units)
c    phi       — within-event standard deviation (ln units)
c    tau       — between-event standard deviation (ln units)
c
c  Reference:
c    Bayless J. and P. Somerville (2024). An Updated Ground Motion
c    Model for Australia Developed Using Broadband Ground Motion
c    Simulations. Proc. AEES 2024 National Conference, Adelaide.
c    Fortran code provided by Jeff Bayless (jeff.bayless@aecom.com).
c =======================================================================


c -----------------------------------------------------------------------
c  SUBROUTINE S24_interp
c
c  Interpolates a GMM coefficient between two tabulated periods.
c  Called internally by bs24_gmm.f for every spectral period that
c  falls between two tabulated values.
c
c  Inputs:
c    per1, per2  — the two bracketing tabulated periods (s)
c    y1,   y2   — coefficient values at per1 and per2
c    perT        — the requested period (s)
c    iflag       — interpolation flag (kept for compatibility)
c
c  Output:
c    yT          — interpolated coefficient value at perT
c
c  Method:
c    Log-linear interpolation in period space:
c      yT = y1 + (y2-y1) * (ln(perT)-ln(per1)) / (ln(per2)-ln(per1))
c    This is the standard approach for GMMs because spectral periods
c    are naturally spaced logarithmically.
c    Exception: if either bracketing period is 0.0 (PGA), fall back
c    to linear interpolation to avoid ln(0).
c -----------------------------------------------------------------------
      subroutine S24_interp (per1, per2, y1, y2, perT, yT, iflag)
      implicit none
      real per1, per2, y1, y2, perT, yT
      integer iflag
      real logper1, logper2, logperT

      if (per1 .eq. 0.0 .or. per2 .eq. 0.0) then
c       Linear interpolation when one end is PGA (period = 0)
        yT = y1 + (y2 - y1) * (perT - per1) / (per2 - per1)
      else
c       Log-linear interpolation for all other periods
        logper1 = alog(per1)
        logper2 = alog(per2)
        logperT = alog(perT)
        yT = y1 + (y2 - y1) * (logperT - logper1) / (logper2 - logper1)
      endif
      return
      end


c -----------------------------------------------------------------------
c  PROGRAM BS24_runner
c
c  Main program. Defines a matrix of test scenarios, calls the BS24
c  subroutine for each one, and writes the results to a CSV file.
c
c  To change the scenarios, edit the data statements below:
c    mags      — moment magnitudes to test
c    rrup_vals — rupture distances (km)
c    vs30_vals — site Vs30 values (m/s)
c    ztor_vals — depth to top of rupture (km)
c    periods   — spectral periods (s); 0.0 = PGA
c    crat_flags— 1 = Cratonic, 0 = NonCratonic
c
c  Fixed rupture geometry used for all scenarios:
c    dip      = 45 degrees (reverse fault)
c    fltWidth = 15 km
c    Rx       = -50 km  (footwall side, so hanging wall term = 0)
c    Ry0      = 0.0 km  (no along-strike offset)
c    z10      = 0.0413 km (default Z1.0 for the reference rock PGA pass)
c    hwflag   = 1 (hanging wall ON; returns 0 for footwall Rx < 0)
c -----------------------------------------------------------------------
      program BS24_runner
      implicit none

c     --- Dimensions of the scenario matrix ---
c     Edit these parameters and the data statements below to change
c     what scenarios are tested.
      integer, parameter :: NMAG  = 5   ! number of magnitudes
      integer, parameter :: NRRUP = 2   ! number of distances
      integer, parameter :: NVS   = 2   ! number of Vs30 values
      integer, parameter :: NZTOR = 2   ! number of Ztor values
      integer, parameter :: NPER  = 22  ! number of spectral periods
      integer, parameter :: NCRAT = 2   ! Cratonic + NonCratonic

c     --- Scenario arrays ---
      real mags(NMAG)
      real rrup_vals(NRRUP)
      real vs30_vals(NVS)
      real ztor_vals(NZTOR)
      real periods(NPER)
      integer crat_flags(NCRAT)

      data mags      / 4.0, 5.0, 6.0, 7.0, 8.0 /
      data rrup_vals / 15.0, 150.0 /
      data vs30_vals / 760.0, 400.0 /
      data ztor_vals / 0.0, 10.0 /
      data periods   /
     &  0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.075,
     &  0.100, 0.150, 0.200, 0.300, 0.400, 0.500, 0.750,
     &  1.000, 1.500, 2.000, 3.000, 4.000, 5.000, 7.500, 10.000 /
      data crat_flags / 1, 0 /

c     --- Fixed rupture geometry variables ---
      real dip        ! fault dip angle (degrees)
      real fltWidth   ! fault down-dip width (km)
      real Rx         ! horizontal distance from fault trace (km)
                      ! negative = footwall; hanging wall term = 0
      real Ry0        ! along-strike distance from rupture end (km)
      real z10        ! Z1.0 basin depth (km)
                      ! 0.0413 is the Fortran default for rock PGA pass
      integer hwflag  ! 1 = compute hanging wall term; 0 = suppress
      integer iflag   ! passed to S24_interp (1 = log interpolation)

c     --- Output variables from BS24 subroutine ---
      real lnY        ! ln(SA) in cm/s/s  — subtract 6.89 to get ln(g)
      real sigma      ! total standard deviation (ln units)
      real phi        ! within-event standard deviation (ln units)
      real tau        ! between-event standard deviation (ln units)
      real period2    ! echo of requested period (returned by subroutine)

c     --- Loop variables ---
      real mag, rrup, vs30, ztor, specT
      integer CratFlag
      integer im, ir, iv, iz, ip, ic

c     --- Set fixed geometry values ---
      dip      = 45.0
      fltWidth = 15.0
      Rx       = -50.0
      Ry0      = 0.0
      z10      = 0.0413
      hwflag   = 1
      iflag    = 1

c     --- Open output CSV file ---
      open(unit=10, file='bs24_fortran_output.csv', status='unknown')
      write(10,'(a)') 'CratFlag,mag,rrup,vs30,ztor,period,'//
     &                'lnY_cmss,sigma,phi,tau'

c     --- Nested loops over all scenario combinations ---
c     Order: CratFlag > magnitude > distance > Vs30 > Ztor > period
      do ic = 1, NCRAT
        CratFlag = crat_flags(ic)   ! 1=Cratonic, 0=NonCratonic

        do im = 1, NMAG
          mag = mags(im)

          do ir = 1, NRRUP
            rrup = rrup_vals(ir)

            do iv = 1, NVS
              vs30 = vs30_vals(iv)

              do iz = 1, NZTOR
                ztor = ztor_vals(iz)

                do ip = 1, NPER
                  specT = periods(ip)

c                 Call the BS24 subroutine (defined in bs24_gmm.f).
c                 Internally it makes two calls to BS_24_sub:
c                   Pass 1: specT=0.01, vs30=760 -> computes pga_rock
c                           (used as input to the nonlinear site term)
c                   Pass 2: specT=requested period, vs30=user vs30
c                           -> computes the final ln(SA)
                  call Bayless_Somerville_2024(
     &              mag, dip, fltWidth, rrup,
     &              vs30, hwflag, lnY, sigma, specT, period2, ztor,
     &              iflag, z10, Rx, Ry0, CratFlag,
     &              phi, tau)

c                 Write results to CSV.
c                 Note: f7.4 for period avoids field overflow at T=10.0
                  write(10, '(i1,a,f4.1,a,f6.1,a,f6.1,a,f5.1,a,'//
     &                        'f7.4,a,f10.5,a,f8.5,a,f8.5,a,f8.5)')
     &              CratFlag, ',', mag,   ',', rrup,  ',',
     &              vs30,     ',', ztor,  ',', specT, ',',
     &              lnY,      ',', sigma, ',', phi,   ',', tau

                enddo  ! period
              enddo    ! ztor
            enddo      ! vs30
          enddo        ! rrup
        enddo          ! magnitude
      enddo            ! CratFlag

      close(10)
      write(*,*) 'Done. Output written to bs24_fortran_output.csv'
      stop
      end
