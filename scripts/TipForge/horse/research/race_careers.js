const RACE_FORM_PAGE_SIZE = 200;

var cached_annual_years = {};
var init_done = false;
var race_form_modal_tab = null;
var scroll_pos = $(window).scrollTop();
var outerChange = true;

var race_form_data = {};
var race_form_modal_yt_links = {};

$(document).ready(function() {
    const num_items = static_stats_table_divs.length;
    let currentFragmentIndex = getCurrentFragment();
    if(currentFragmentIndex){
        goto_slide = currentFragmentIndex;
    }

    setupFragmentRuntimeHandling();
    setup_years_stat_dropdown_menus();
    setup_annual_stat_dropdown_menus();

    complexLoader(slide_loader_map[currentFragmentIndex]);
    setupLoading(null);
    setupSlider(num_items);

    locale_change_callbacks.push(setup_years_datepicker);

    init_race_form_modal_tabulator();

    document.getElementById("content-r").classList.remove("hidden");

    $('.slider').slick('setPosition');

    init_done = true;
});

function setupLoading(loader) {
    $('.slider').on('beforeChange', function(event, slick, currentSlide, nextSlide){
        setFragmentIdentifier(nextSlide);
        complexLoader(slide_loader_map[nextSlide]);
    });
}

function complexLoader(index, force_reload=false)
{
    $.each(static_stats_table_divs[index], function(k, v){ 
            if(tables[k] == null || force_reload){
                if(v['url'] == undefined){
                    var table = _loader_static(index, k, v);
                }
                else{
                    var table = _loader_dynamic(index, k, get_url(index), v);
                }

                // Because tabulator loading is weird
                $(`#${k}`).ready(function(){
                    table.redraw(true);
                });

                if (tables[k] == undefined){
                    tables.push(table);
                }
                tables[k] = table;
            }else{ 
                // see if the locale of the table matches the currently selected one, if not, redraw
                if(tables[k].getLocale() != current_locale) {
                  tables[k].setLocale(current_locale);
                  tables[k].redraw(true);
                }
            }
    });
}

function _get_cache_identifier(index, div, url)
{
    return index + div + url.params.join('_');
}

function _loader_dynamic(index, div, url, value)
{
    let cache_identifier  = _get_cache_identifier(index, div, url)

    value['url'] = url;


    let tabs = Tabulator.prototype.findTable("#"+div);

    if (tabs != false){
        tabs[0].destroy();
    }


    if(cache_identifier in cached_dyn_table_data && index != 'annual_stat')
    {

        return new Tabulator("#"+div, {
            layout: "fitDataFill",
            responsiveLayout: "collapse",
            columns: cached_dyn_table_data[cache_identifier].columns_def,
            data: cached_dyn_table_data[cache_identifier].table_data,
            langs: {
                "en-gb": dictionary["en-gb"],
                "hu-hu": dictionary["hu-hu"],
            },
            locale: current_locale,
            responsiveLayoutCollapseStartOpen:false,
            renderComplete:function(){
            },
            rowFormatter:function(row){
                if(row.getNextRow() == false){
                        row.getElement().style.fontWeight = "bold";
                }
            },
        });
    }
    else
    {
        return new Tabulator("#"+div, {
            layout: "fitDataFill",
            responsiveLayout: "collapse",
            columns: race_career_tables[index](value),
            langs: {
                "en-gb": dictionary["en-gb"],
                "hu-hu": dictionary["hu-hu"],
            },
            locale: current_locale,
            ajaxURL:url.url,
            responsiveLayoutCollapseStartOpen:false,
            ajaxResponse:function(url, params, response){
                let table_data = process_response(index, response, value);
                cached_dyn_table_data[cache_identifier] = {
                    'table_data': table_data,
                    'columns_def': race_career_tables[index](value)
                }
                return table_data;
            },
            ...(index == 'annual_stat' ? {
                ajaxRequestFunc: customLoaderPromiseAnnualStat,
            }: {}),
            dataLoaded:function(data){
                $(".slider-for").slick('refresh');
            },
            renderComplete:function(){
                if(index == 'annual_stat'){
                    setupFilteredRaceForms();
                }
            },
            rowFormatter:function(row){
                if(row.getNextRow() == false){
                        row.getElement().style.fontWeight = "bold";
                }
            },
        });
    }
}

function _loader_static(index, div, data_source, param=null)
{
    let progeny_stat_grouping_options = 'progeny_stat' == index ? {
        groupBy: 'born',
        groupHeader: function(value, count, data, group){
            return value + "<span>(" + count + ")</span>";
        },
        groupVisibilityChanged:function(group, visible){
            let table = group.getTable();
            let groups = table.getGroups();
            let all_group_closed = !groups.map(g => g.isVisible()).includes(true);

            if (all_group_closed == true){
                table.rowManager.tableElement.style.minWidth = "";
            }
        },
    }:{};


    let table_data = 'race_form_stat' == index ? {
        pagination: 'remote',
        ajaxURL: get_url(index).url,
        ajaxParams:{
            limit: RACE_FORM_PAGE_SIZE,
        },
        paginationDataReceived:{
            "data":"results",
        },
        dataLoaded:function(data){
            $(".slider-for").slick('refresh');
        },
        ajaxRequestFunc: customLoaderPromise,
    }: {
        data: data_source.data,
    };

    return new Tabulator("#"+div, {
        layout: "fitDataFill",
        responsiveLayout: "collapse",
        cellVertAlign:"middle",
        columns: race_career_tables[index](param),
        langs: {
            "en-gb": dictionary["en-gb"],
            "hu-hu": dictionary["hu-hu"],
        },
        locale: current_locale,
        ...table_data,
        responsiveLayoutCollapseStartOpen:false,
        rowFormatter:function(row){
            if (!['race_form_stat', 'progeny_stat'].includes(index)){
                if(row.getNextRow() == false){
                        row.getElement().style.fontWeight = "bold";
                }
            }
        },
        ...progeny_stat_grouping_options,
    });
}

function get_url(index)
{
    let base_url = api_urls[index];
    var url = {url: '', params: [], params_dict: {}};

    if(index == 'years_stat')
    {
        let loc = $("#dropdown-menu-link-year-" + index).attr('location');

        url.url = base_url + loc + "/";

        url.url = [base_url, loc, entity_id].join('/');
        url.params.push(loc);
        url.params.push(entity_id);
        
        if(discipline == 'gallop' && ['driver_jockey', 'trainer'].includes(entity_type))
        {
            let race_type = $(".dropdown #dropdown-menu-link-type-sel-years_stat").attr('race_type');

            url.url += '/' + race_type;
            url.params.push(race_type)
        } 
    }
    else if(index == 'annual_stat')
    {
        let year = $('#racing-years-datepicker').datepicker('getDate').getFullYear();
        let loc = $(".dropdown #dropdown-menu-link-loc-annual_stat").attr('location');
        let grouping = $(".dropdown #dropdown-menu-link-grouping-annual_stat").attr('grouping');

        url.params_dict['grouping'] = grouping;
        url.params_dict['location'] = loc;
        url.params_dict['year'] = year;
        

        if(discipline == 'gallop' && ['driver_jockey', 'trainer'].includes(entity_type))
        {
            let race_type = $(".dropdown #dropdown-menu-link-type-sel-annual_stat").attr('race_type');

            url.url = [base_url, loc, entity_id, race_type, grouping, year].join('/');
            url.params = url.params.concat([loc, entity_id, race_type, grouping, year]);
            url.params_dict['race_type'] = race_type;
        } 
        else{
            url.url = [base_url, loc, entity_id, grouping, year].join('/');
            url.params = url.params.concat([loc, entity_id, grouping, year]);
        }

        var grouping_table_header = grouping;

        if(grouping == 'monthly'){
            grouping_table_header = 'month';
        }
        else if(grouping == 'driver_jockey'){
            if(discipline == 'trotting'){
                grouping_table_header = 'driver';
            }
            else{
                grouping_table_header = 'jockey';
            }
        }
        url.params_dict['grouping_table_header'] = grouping_table_header;

    }else if (index == 'race_form_stat'){
        url.url = [base_url, entity_id].join('/');
    }
    return url;
}

function process_response(index, response, value)
{
    var tab_result = [];

    if(index == 'years_stat')
    {
        var response_data = response.yearly_results;

        if(response.yearly_results){
            response_data = response_data.reverse();
        }

        $.each(response_data, function(){ 
            tab_result.push(map_results(this));
        });

        if(response.total_results){
            total_results = map_results(response.total_results);
            total_results.year = 'total_results';
            tab_result.push(total_results);
        }
    }
    else if(index == 'annual_stat')
    {
        var sort_by_group = 'total_prize';
        var asc_order = false;

        $.each(response.partial_results, function(k, v){ 
            tab_result.push(map_results_annual(this, k, v, value));
        });

        /*Sort by grouping*/
        if(value.url.params_dict.grouping == 'monthly' || value.url.params_dict.grouping == 'distance'){
            sort_by_group = value.url.params_dict.grouping_table_header;
            asc_order = true;
        }

        tab_result = sort_data_by_group(tab_result, sort_by_group, asc_order);

        if(response.total_results){
            tab_result.push(map_results_annual(response.total_results, 'total_results', null, value));
        }

    }
    return tab_result;
}


function map_results(item)
{
    return {
        "year": item["year"],
        "runs": item["race_count"],
        "1st": item["results"]["1st"],
        "2nd": item["results"]["2nd"],
        "3rd": item["results"]["3rd"],
        "4th_5th": item["results"]["4th"] + item["results"]["5th"],
        "4th": item["results"]["4th"],
        "5th": item["results"]["5th"],
        "6th": item["results"]["6th"],
        "unplaced": item["results"]["unplaced"],
        "total_prize": item["total_prize"],
    }
}

function map_results_annual(item, k, v, value)
{
    let response = {
        "id": item['id'],
        "runs": item["race_count"],
        "1st": item["1st"],
        "1st_percentage": item["1st_percentage"],
        "2nd": item["2nd"],
        "3rd": item["3rd"],
        "4th_5th":  item["4th"] + item["5th"],
        "4th": item["4th"],
        "5th": item["5th"],
        "6th": item["6th"],
        "total_prize": item["total_prize"],
        "races": item['races'],
        "short_name": item['short_name'] || '',
    }
    response[value.url.params_dict.grouping_table_header] = k;
    return response;
}

function inject_race_video_links(response, tabulator)
{
    let race_ids = [... new Set(response.results.map(x=>x.race_id))];
    $.get(`${api_urls['yt_media_api']}/${race_ids.join(',')}`, function(data) {
        $(response.results.map(x=>x)).each(function(idx, content){
            if(content.race_id in data.videos && data.videos[content.race_id] != null){
                response.results[idx].video_link = `https://youtube.com/watch?v=${data.videos[content.race_id]}`;
            }else{
                response.results[idx].video_link = null;
            }
        });
        tabulator.setData(response.results);
    });
}

function setup_years_stat_dropdown_menus()
{
  $(".dropdown-menu-years-loc").on('click', 'a', function(){
      let btn_orig = $(".dropdown #dropdown-menu-link-year-years_stat");
      $(btn_orig).html($(this).html());
      $(btn_orig).attr('location', $(this).attr('location'));
      complexLoader('years_stat', force_reload = true);
 });
 $(".dropdown-menu-years-type-sel").on('click', 'a', function(){
    let btn_orig = $(".dropdown #dropdown-menu-link-type-sel-years_stat");

    $(btn_orig).html($(this).html());
    $(btn_orig).attr('race_type', $(this).attr('race_type'));

    complexLoader('years_stat', force_reload = true);
});
}

function setup_years_datepicker()
{
    if ( $('#racing-years-datepicker').length ) {
        $('#racing-years-datepicker').datepicker('destroy');
      }

      if (typeof racing_years != "undefined") {
        $('#racing-years-datepicker').datepicker({
          format: " yyyy",
          language: current_locale,
          viewMode: "years",
          minViewMode: "years",
          maxViewMode: "years",
          endDate: '+0y',
          autoclose: true,
          beforeShowYear: function(d){
            return (racing_years.includes(d.getFullYear()));
          },
        }).on('changeDate', function(e){
            complexLoader('annual_stat', force_reload = true);
        });
    }
}

function setup_annual_stat_dropdown_menus()
{
    setup_years_datepicker();

    $(".dropdown-menu-annual-loc").on('click', 'a', function(){
        let btn_orig = $(".dropdown #dropdown-menu-link-loc-annual_stat");
        $(btn_orig).html($(this).html());
        $(btn_orig).attr('location', $(this).attr('location'));
        complexLoader('annual_stat', force_reload = true);
    });
    $(".dropdown-menu-annual-type-sel").on('click', 'a', function(){
        let btn_orig = $(".dropdown #dropdown-menu-link-type-sel-annual_stat");
    
        $(btn_orig).html($(this).html());
        $(btn_orig).attr('race_type', $(this).attr('race_type'));
    
        complexLoader('annual_stat', force_reload = true);
    });
    $(".dropdown-menu-annual-grouping").on('click', 'a', function(){
        let btn_orig = $(".dropdown #dropdown-menu-link-grouping-annual_stat");
    
        $(btn_orig).html($(this).html());
        $(btn_orig).attr('grouping', $(this).attr('grouping'));

        // Change grouping name in the header
        let en_text = $('#annual_stat_grouping > h2 > span.trans-text-en');
        let hu_text = $('#annual_stat_grouping > h2 > span.trans-text-hu');
        $(en_text).text(capitalize(dictionary["en-gb"]["general"][$(this).attr('grouping')]));
        $(hu_text).text(capitalize(dictionary["hu-hu"]["general"][$(this).attr('grouping')]));
    
        complexLoader('annual_stat', force_reload = true);
    });
}

function load_race_form_for_group(races)
{
    let identifier = getRaceFormDatasetIdentifier();

    if(identifier in race_form_data)
    {
        let filtered_races = [];
        $(race_form_data[identifier].results).each(function(index, e) {
            if(races.includes(e.form_id)){
                e.id = index;
                filtered_races.push(e);
            }
        });
        scroll_pos = $(window).scrollTop();

        $('#race_form_modal').modal('show');

        race_form_modal_tab.setData(filtered_races).then(function(){
            race_form_modal_tab.setPage(1);
        });
    }
}


function init_race_form_modal_tabulator()
{
    $('#race_form_modal').on('shown.bs.modal', function (e) {
        $("#race_form_modal_body .tabulator-responsive-collapse-toggle").trigger('click');
        race_form_modal_tab.redraw(true);
    });

    $('#race_form_modal').on('hide.bs.modal', function (e) {
        $(window).scrollTop(scroll_pos);
    });

    race_form_modal_tab =  new Tabulator("#race_form_modal_body", {
        layout: "fitDataFill",
        responsiveLayout: "collapse",
        columns: race_career_tables['race_form_stat'](null),
        langs: {
            "en-gb": dictionary["en-gb"],
            "hu-hu": dictionary["hu-hu"],
        },
        locale: current_locale,
        pagination: "local",
        paginationSize: RACE_FORM_PAGE_SIZE,
        index: "id",
        pageLoaded:function(pageno){
            let table = this;
            let data = table.getData().map(x=>x);
            table.modules.ajax.showLoader();
            fetchYouTubeLinks(data, pageno, RACE_FORM_PAGE_SIZE).then(function(races){
                table.updateData(races).then(function(){
                    table.modules.ajax.hideLoader();
                }).catch(function(error){
                    table.modules.ajax.hideLoader();
                });
            });
        },
    });
    tables.push(race_form_modal_tab);
}

function setupFilteredRaceForms()
{
    $('.filtered-race-forms').off('click').on('click', function(){
        /*Retrieve tabulator component from a row position data*/
        let rowPosition = $(this).attr('position');
        let [table_div] = Object.keys(static_stats_table_divs[slide_loader_map[$('.slider').slick('slickCurrentSlide')]]);
        let [table] = Tabulator.prototype.findTable(`#${table_div}`);
        let row = table.getRowFromPosition(rowPosition);
        /*Runs field required for the original cell click handler*/
        let cell = row.getCell('runs');
        annualStatRaceFormCellClick(null, cell);
    });
}

function annualStatRaceFormCellClick(e, cell){
    if(cell.getValue() != '0'){
        var race_ids = [];
        if (cell.getRow().getNextRow() == false){
            let rows = cell.getTable().getData();
            // pop sum row
            rows.pop();

            rows.forEach(function(row) {
                row.races.forEach(function(e) {
                    race_ids.push(e.form_id);
                });
            });
        }else{
            cell.getData().races.forEach(function(e) {
                race_ids.push(e.form_id);
            });
        }
        load_race_form_for_group(race_ids);
    }
}

function setupFragmentRuntimeHandling(){
    window.onhashchange = function(){
        if(outerChange){
            setTimeout(function() {
                $('.slider-for').slick('slickGoTo', getCurrentFragment());
                $('.slider').slick('setPosition');
              }, 100);
        }else{
            outerChange = true;    
        }
    };
}

function setFragmentIdentifier(currentSlideIndex){
    if(currentSlideIndex != getCurrentFragment()){
        window.location.hash = slide_loader_map[currentSlideIndex];
        outerChange = false;
    }
}

function getCurrentFragment(){
    if(window.location.hash) {
        let hash = window.location.hash.substring(1);
        let fragmentIndex = getKeyByValue(slide_loader_map, hash);

        if (fragmentIndex){
            return parseInt(fragmentIndex);
        }
    }
    // missing or invalid fragment identifier
    return 0;
}

const getKeyByValue = (obj, value) => 
        Object.keys(obj).find(key => obj[key] === value);


function customLoaderPromiseAnnualStat(url, config, params)
{
    var self = this;

    return new Promise(function (resolve, reject) {

        //set url
        url = self.urlGenerator.call(self.table, url, config, params);

        if (url) {
            config.headers = {};
            //send request
            fetch(url, config).then(function (response) {
                if (response.ok) {
                    response.json().then(function (data) {

                        let annual_stat_params = get_url("annual_stat").params_dict;

                        let dataset_id = getRaceFormDatasetIdentifier();

                        if (dataset_id in race_form_data){
                            resolve(data);
                        }else{
                            fetch(get_url("race_form_stat").url + "?" + new URLSearchParams({
                                location: annual_stat_params['location'],
                                year: annual_stat_params['year'],
                                ...(annual_stat_params['race_type'] != undefined ? {race_type: annual_stat_params['race_type']}: {}),
                            }), config).then(function (race_form_response) {
                                if (race_form_response.ok) {
                                    race_form_response.json().then(function (race_form_data_response) {
                                        race_form_data[dataset_id] = race_form_data_response;
                                        resolve(data);
                                    });
                                }else{
                                    resolve(data); 
                                }
                            }).catch(function (error) {
                                resolve(data);
                            });
                        }
                    }).catch(function (error) {
                        reject(error);
                        console.warn("Ajax Load Error - Invalid JSON returned", error);
                    });
                } else {
                    console.error("Ajax Load Error - Connection Error: " + response.status, response.statusText);
                    reject(response);
                }
            }).catch(function (error) {
                console.error("Ajax Load Error - Connection Error: ", error);
                reject(error);
            });
        } else {
            console.warn("Ajax Load Error - No URL Set");
            resolve([]);
        }
    });
}

function customLoaderPromise(url, config, params) {
    var self = this;

    return new Promise(function (resolve, reject) {

        //set url
        url = self.urlGenerator.call(self.table, url, config, params);

        if (url) {
            config.headers = {};
            //send request
            fetch(url, config).then(function (response) {
                if (response.ok) {
                    response.json().then(function (data) {

                        filter_invalid_race_form_fields(data);

                        // Fetch video links from media API
                        let race_ids = [... new Set(data.results.map(x=>x.race_id))];
                        
                        if (race_ids.length > 0){
                            fetch(`${api_urls['yt_media_api']}/${race_ids.join(',')}`, config).then(function (video_response) {
                                if (video_response.ok) {
                                    video_response.json().then(function (video_data) {
                                        // Update video links
                                        $(data.results.map(x=>x)).each(function(idx, content){
                                            if(content.race_id in video_data.videos && video_data.videos[content.race_id] != null){
                                                data.results[idx].video_link = `https://youtube.com/watch?v=${video_data.videos[content.race_id]}`;
                                            }else{
                                                data.results[idx].video_link = null;
                                            }
                                        });
                                        resolve(data);
                                    });
                                }else{
                                    resolve(data); 
                                }
                            }).catch(function (error) {
                                resolve(data);
                            });
                        }else{
                            resolve(data);
                        }
                    }).catch(function (error) {
                        reject(error);
                        console.warn("Ajax Load Error - Invalid JSON returned", error);
                    });
                } else {
                    console.error("Ajax Load Error - Connection Error: " + response.status, response.statusText);
                    reject(response);
                }
            }).catch(function (error) {
                console.error("Ajax Load Error - Connection Error: ", error);
                reject(error);
            });
        } else {
            console.warn("Ajax Load Error - No URL Set");
            resolve([]);
        }
    });
};

function filter_invalid_race_form_fields(data)
{
    //Remove 0:00:00 km time for trotting
    if (discipline == 'trotting'){
        $(data.results.map(x=>x)).each(function(idx, content){
            if(content.km_time == '0:00.0'){
                data.results[idx].km_time = '-';
            }
        });
    }
}

function getRaceFormDatasetIdentifier()
{
    let annual_stat_params = get_url("annual_stat").params_dict;
    return `${annual_stat_params['location']}_${annual_stat_params['year']}_${annual_stat_params['race_type']}`;
}

function fetchYouTubeLinks(races, page, limit)
{
        return new Promise(function (resolve, reject) {
            if (!races.length){
                resolve(races);
            }else{
                // lets load the first page
                let race_ids = [... new Set(races.slice((page-1) * limit, limit * page).filter(value => value.video_link == undefined).map(x=>x.race_id))];

                if(!race_ids.length){
                    resolve(races);
                }else{
                    fetch(`${api_urls['yt_media_api']}/${race_ids.join(',')}`).then(function (video_response) {
                        if (video_response.ok) {
                            video_response.json().then(function (video_data) {
                                // Update video links
                                $(races.map(x=>x)).each(function(idx, content){
                                    if(content.race_id in video_data.videos && video_data.videos[content.race_id] != null){
                                        races[idx].video_link = `https://youtube.com/watch?v=${video_data.videos[content.race_id]}`;
                                        races[idx].video = true;
                                    }
                                });
                                resolve(races);
                            });
                        }else{
                            //todo error message
                            resolve(races);
                        }
                    }).catch(function (error) {
                        //todo error message
                        resolve(races);
                    });
                }
            }
        });
}