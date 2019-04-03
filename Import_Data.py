""" This script gets data from pymortar and xbos_services_getter. """

import pymortar
import xbos_services_getter
import pandas as pd
from dateutil.parser import parse
from datetime import datetime, timedelta
from pytz import timezone
from collections import defaultdict


class Import_Data():

    """ This class queries data from Mortar and xbos_services_getter.

    Note
    ----
    For pymortar, set the evironment variables - $MORTAR_API_USERNAME & $MORTAR_API_PASSWORD.

    For Mac,
    1. vi ~/.bash_profile
    2. Add at the end of file,
        1. export $MORTAR_API_USERNAME=username
        2. export $MORTAR_API_PASSWORD=password
    3. source ~/.bash_profile


    """

    def __init__(self):
        """ Constructor. """

        self.client = pymortar.Client({})


    @staticmethod
    def convert_to_utc(time):
        """ Convert time to UTC

        Parameters
        ----------
        time    : str
            Time to convert. Has to be of the format '2016-01-01T00:00:00-08:00'.

        Returns
        -------
        str
            UTC timestamp.

        """

        # time is already in UTC
        if 'Z' in time:
            return time
        else:
            time_formatted = time[:-3] + time[-2:]
            dt = datetime.strptime(time_formatted, '%Y-%m-%dT%H:%M:%S%z')
            dt = dt.astimezone(timezone('UTC'))
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    def get_data(self, sites, start, end, point_type, aggregate, window):
        """

        To Do
        -----
        1. get_weather_data can't retrieve > 3 months data.

        Parameters
        ----------
        sites            : list(str)
            List of sites.
        start           : str
            Start date. Format - 'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DDTHH:MM:SSZ' where Z means UTC.
        end             : str
            End date. Format - 'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DDTHH:MM:SSZ' where Z means UTC.
        point_type      : str
            Type of data, i.e. Green_Button_Meter, Building_Electric_Meter...
        aggregate       : pymortar aggregation object
            Values include pymortar.MEAN, pymortar.MAX, pymortar.MIN,
        pymortar.COUNT, pymortar.SUM, pymortar.RAW (the temporal window parameter is ignored)
        window          : str
            Size of the moving window. e.g. '15m', '1d'...

        Returns
        -------
        str, pd.DataFrame()
            Site name and dataframe containing meter, weather & tstat data for the site.

        """

        # CHECK: Hacky code. Change later.
        for key, value in aggregate.items():
            if value == 'MEAN':
                aggregate[key] = pymortar.MEAN
            elif value == 'MAX':
                aggregate[key] = pymortar.MAX

        for site in sites:

            df_meter, map_uuid_meter = self.get_meter_data(site=[site], start=start, end=end,
                                                           point_type=point_type,
                                                           window=window['meter'])
            # CHECK: Hacky code. Change later
            df_meter.columns = ['power']

            df_oat = self.get_weather_data(site=site, start=start, end=end, window=window['tstat'])
            # CHECK: Hacky code. Change later
            df_oat.columns = ['oat']
            df_oat.index = df_oat.index.tz_localize('UTC')

            df_tstat = self.get_tstat_data(site=[site], start=start, end=end,
                                           window=window['tstat'])
        
            df = pd.concat([df_meter, df_oat, df_tstat], axis=1, join='outer').dropna()

            yield site, df

    def get_meter_data(self, site, start, end, point_type="Green_Button_Meter", agg='MEAN', window='15m'):
        """ Get meter data from Mortar.

        Parameters
        ----------
        site            : list(str)
            List of sites.
        start           : str
            Start date - 'YYYY-MM-DDTHH:MM:SSZ'
        end             : str
            End date - 'YYYY-MM-DDTHH:MM:SSZ'
        point_type      : str
            Type of data, i.e. Green_Button_Meter, Building_Electric_Meter...
        agg             : pymortar aggregation object
            Values include pymortar.MEAN, pymortar.MAX, pymortar.MIN, 
        pymortar.COUNT, pymortar.SUM, pymortar.RAW (the temporal window parameter is ignored)
        window          : str
            Size of the moving window.
        
        Returns
        -------
        pd.DataFrame(), defaultdict(list)
            Meter data, dictionary that maps meter data's columns (uuid's) to sitenames.

        """

        # CHECK: Hacky code. Change this later
        if agg == 'MEAN':
            agg = pymortar.MEAN

        # CHECK: Does Mortar take in UTC or local time? 
        # Convert time to UTC
        start = self.convert_to_utc(start)
        end = self.convert_to_utc(end)

        query_meter = "SELECT ?meter WHERE { ?meter rdf:type brick:" + point_type + " };"

        # Define the view of meters (metadata)
        meter = pymortar.View(
            name="view_meter",
            sites=site,
            definition=query_meter
        )

        # Define the meter timeseries stream
        data_view_meter = pymortar.DataFrame(
            name="data_meter", # dataframe column name
            aggregation=agg,
            window=window,
            timeseries=[
                pymortar.Timeseries(
                    view="view_meter",
                    dataVars=["?meter"]
                )
            ]
        )

        # Define timeframe
        time_params = pymortar.TimeParams(
            start=start,
            end=end
        )

        # Form the full request object
        request = pymortar.FetchRequest(
            sites=site,
            views=[meter],
            dataFrames=[data_view_meter],
            time=time_params
        )

        # Fetch data from request
        response = self.client.fetch(request)

        # resp_meter = (url, uuid, sitename)
        resp_meter = response.query('select * from view_meter')

        # Map's uuid's to the site names
        map_uuid_sitename = defaultdict(list)
        for (url, uuid, sitename) in resp_meter:
            map_uuid_sitename[uuid].append(sitename)

        return response['data_meter'], map_uuid_sitename

    def get_weather_data(self, site, start, end, window):
        """ This functions retrieves OAT data from xbos_services_getter.

        To Do
        -----
        1. Chunk data if query time > 3months

        Parameters
        ----------
        sitename    : list(str)
            List of buildings.
        start       : str
            Start time of data. Format - '%Y-%m-%dT%H:%M:%SZ'
        end         : str
            End time of data. Format - '%Y-%m-%dT%H:%M:%SZ'
        window      : str
            Data interval. e.g. '15m', '1d'...

        Returns
        -------
        pd.DataFrame()
            OAT data.

        """

        # CHECK: Does Mortar take in UTC or local time?
        # Convert time to UTC
        start = self.convert_to_utc(start)
        end = self.convert_to_utc(end)

        # Convert start and end from str to datetime objects
        # Note: Input is str for uniformity and converting to datetime because of xbos_services_getter
        start = parse(start)
        end = parse(end)

        # start = datetime(year=2018, month=1, day=1, hour=0, minute=0).replace(tzinfo=pytz.utc)
        # end = start + datetime.timedelta(days=7)

        outdoor_historic_stub = xbos_services_getter.get_outdoor_historic_stub()
        data = xbos_services_getter.get_outdoor_temperature_historic(outdoor_historic_stub=outdoor_historic_stub,
                                                                     start=start, end=end,
                                                                     building=site, window=window)

        # Convert index to tz_naive so that it can be joined with meter data and oat data
        data = data.tz_localize(None)

        return pd.DataFrame(data)

    def get_tstat_data(self, site, start, end, agg=pymortar.MAX, window='1m'):
        """ Get tstat data from Mortar.

        Parameters
        ----------
        site            : list(str)
            List of sites.
        start           : str
            Start date - 'YYYY-MM-DDTHH:MM:SSZ'
        end             : str
            End date - 'YYYY-MM-DDTHH:MM:SSZ'
        agg             : pymortar aggregation object
            Values include pymortar.MEAN, pymortar.MAX, pymortar.MIN, 
        pymortar.COUNT, pymortar.SUM, pymortar.RAW (the temporal window parameter is ignored)
        window          : str
            Size of the moving window.
        
        Returns
        -------
        pd.DataFrame()
            Dataframe containing tstat data for all sites.

        """

        # CHECK: Does Mortar take in UTC or local time? 
        # Convert time to UTC
        start = self.convert_to_utc(start)
        end = self.convert_to_utc(end)

        query_tstat = "SELECT ?tstat ?room ?zone ?state ?temp WHERE { \
            ?tstat bf:hasLocation ?room . \
            ?zone bf:hasPart ?room . \
            ?tstat bf:hasPoint ?state . \
            ?tstat bf:hasPoint ?temp . \
            ?zone rdf:type/rdfs:subClassOf* brick:Zone . \
            ?tstat rdf:type/rdfs:subClassOf* brick:Thermostat . \
            ?state rdf:type/rdfs:subClassOf* brick:Thermostat_Status . \
            ?temp  rdf:type/rdfs:subClassOf* brick:Temperature_Sensor  . \
        };"

        # Define the view of tstat (metadata)
        tstat = pymortar.View(
            name="view_tstat",
            sites=site,
            definition=query_tstat
        )

        # Define the meter timeseries stream
        data_view_tstat = pymortar.DataFrame(
            name="data_tstat", # dataframe column name
            aggregation=agg,
            window=window,
            timeseries=[
                pymortar.Timeseries(
                    view="view_tstat",
                    dataVars=["?state", "?temp"]
                )
            ]
        )

        # Define timeframe
        time_params = pymortar.TimeParams(
            start=start,
            end=end
        )

        # Form the full request object
        request = pymortar.FetchRequest(
            sites=site,
            views=[tstat],
            dataFrames=[data_view_tstat],
            time=time_params
        )

        # Fetch data from request
        response = self.client.fetch(request)

        # Final dataframe containing all sites' data
        df_result = pd.DataFrame()
        
        tstat_df = response['data_tstat']
        tstats = [tstat[0] for tstat in response.query("select tstat from view_tstat")]
        error_df_list = []

        for i, tstat in enumerate(tstats):

            q = """
                SELECT state_uuid, temp_uuid, room, zone, site
                FROM view_tstat
                WHERE tstat = "{0}";
            """.format(tstat)
        
            res = response.query(q)
            if not res:
                continue

            state_col, iat_col, room, zone, site = res[0]
            df = tstat_df[[state_col, iat_col]]
            
            # A single site has many tstat points. Adding site+str(i) distinguishes each of them.
            # CHECK: This can have a better naming scheme.
            df.columns = ['s'+str(i), 't'+str(i)]

            df_result = df_result.join(df, how='outer')

        return df_result

    def get_error_message(self, x, resample_minutes=60):
        """ Creates error message for a row of error_df (get_tstat())

        Parameters
        ----------
        x                   : row of pd.DataFrame()
            Pandas row.
        resample_minutes    : int
            Resampling minutes.
        
        Returns
        -------
        str
            Error message.

        """
        
        dt_format = "%Y-%m-%d %H:%M:%S"
        st = x.name
        st_str = st.strftime(dt_format)
        et_str = (st + timedelta(minutes=resample_minutes)).strftime(dt_format)
        site = x.site
        room = x.room
        zone = x.zone
        heat_percent = round(x.heat_percent, 2)
        cool_percent = round(x.cool_percent, 2)
        msg = "From {0} to {1}, zone: \'{2}\' in room: \'{3}\' at site: \'{4}\', " \
              "was heating for {5}% of the time and cooling for {6}% of the time".format(
            st_str, et_str, zone, room, site, heat_percent, cool_percent
        )

        return msg
