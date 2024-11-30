# crawl-apnic-aspop

This script will crawl the longitudinal data for the
[APNIC Customers per AS Measurements](https://stats.labs.apnic.net/aspop).

## Dependencies

* Bash (tested on version 5.2+)
* Python 3
* [`just`](https://github.com/casey/just/) (version 1.23.0+)
* `curl`
* GNU date (macOS users will need to install `coreutils`)
* GNU parallel

## Usage (ad-hoc, one-off)

To crawl one day, e.g., `2024-01-01`:

```bash
cd src/
bash get-date.sh 2024-01-01
```

The output will be saved to `data/aspop.2024-01-01.csv`.

If there are no input parameters, the script will download the most recent day,
which we define as `TODAY - 2 days`.

It's helpful to store log data about when and how the data was downloaded,
especially in case of errors. The following will store a logfile next to the
.csv output (this is done by default when crawling):

```bash
cd src/
mkdir -p ../data
bash get-date.sh 2024-01-01 | tee -a ../data/aspop.2024-01-01.log
```

## Usage (crawl entire dataset)

> [!TIP]
> We recommend crawling a couple days of data to start off with as part of your
> initial analysis before crawling the entire dataset (11+ years, 5.0GB
> uncompressed).

To crawl the entire dataset to `data/`:

```bash
cd src/
just crawl-all
```

Log files for each day accompany each .csv, e.g., `data/aspop.2024-01-01.txt`.

## Usage (crawl subsequent updates)

To update an existing crawl, you can do one of the following:
* [Update the missing entries (ad-hoc)](#update-dataset)
* [Setup a cronjob to update daily](#cronjob)

### Update Dataset

We expect all data files (e.g., `aspop.2024-01-01.csv`) to be in `data/`.
The following will compute the missing days and crawl all missing days:

```bash
cd src/
just crawl-update
```

### Cronjob

Set up the following line in your crontab (e.g., via `crontab -e`) to update
the dataset daily at 5am and log output to `~/crawl.log.txt`:

```
0 5 * * * just --justfile /path/to/unboxing_apnic/data/crawl-apnic-aspop/src/justfile crawl-update | tee -a ~/crawl.log.txt
```

## LICENSE

Code in this directory is licensed [`MIT`](./LICENSE).

The APNIC Customers per AS Measurements data is copyright APNIC, and, as
of writing, does not have a specified license.
