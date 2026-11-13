# ITI AIA auto-calibration table

`sdo_autocal_table.csv` is an unmodified snapshot of the table used by
InstrumentToInstrument (ITI) for AIA degradation correction.

- Original URL: <http://kanzelhohe.uni-graz.at/iti/sdo_autocal_table.csv>
- Retrieved via HTTPS: <https://kanzelhohe.uni-graz.at/iti/sdo_autocal_table.csv>
- Verified on: 2026-11-12
- Upstream repository: <https://github.com/RobertJaro/InstrumentToInstrument>
  (`dataset/sdo_autocal_table.csv`)
- SHA-256: `7467f9fb65c2448237312d3973583351774bd2f2ac1da9eb564b0cfc75993fa3`
- Size: 95,141 bytes; 589 rows.
- Date coverage: 2010-05-15 00:01:06.184 through 2020-07-08 00:01:09.184.
- Channels: 94, 131, 171, 193, 211, 304, and 335 Angstrom.


FOXES prefers a valid `~/.iti/sdo_autocal_table.csv` to preserve existing
calibration choices, and falls back to this snapshot if the cache is missing,
unreadable, or invalid. It does not download or write a calibration cache.
ITI's nearest-row correction is unchanged, including selection of the endpoint
row for observations outside the table's date range. Bundling this snapshot
does not extend its calibration coverage.

To update the snapshot, obtain the upstream CSV, validate it with
`data.aia_calibration._read_table`, and update the checksum and coverage here
together with the snapshot integrity test in `tests/test_aia_calibration.py`.
