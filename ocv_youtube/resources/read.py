import youtube_dl

def readvideo(url):
    #-- Try read video streaming
    video_url = url
    ydl_opts = {}
    # create youtube-dl object
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    # set video url, extract video information
    info_dict = ydl.extract_info(video_url, download=False)
    # get video formats available
    formats = info_dict.get('formats',None)
    title = info_dict.get('title',None)
    for f in formats:
        # Set resolution video
        if f.get('format_note',None) == '480p':
            print('titulo del video:',title,'\ncalidad:',f.get('format_note', None))
            return f.get('url',None)
