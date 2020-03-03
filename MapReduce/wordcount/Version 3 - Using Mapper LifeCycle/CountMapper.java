package cps534.count_v3;

import java.io.IOException;
import java.util.HashMap;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class CountMapper extends Mapper<Object, Text, Text, IntWritable>
{
	private final static IntWritable one = new IntWritable(1);
	private Text word = new Text();
	
	private HashMap<String, Integer> hashMap;
	
	
	protected void setup(Context context) throws IOException, InterruptedException 
	{
		hashMap = new HashMap<String, Integer>();
	}
	
	
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException 
	{
		StringTokenizer itr = new StringTokenizer(value.toString());
        while(itr.hasMoreElements())
        {
            String string = itr.nextToken();
            Integer count = hashMap.get(string);
            
            if(count == null)
            {
            	count = new Integer(0);
           	}
            
            count += 1;
            
            hashMap.put(string,count);       
        }
	}
        
	protected void cleanup(Context context) throws IOException, InterruptedException 
	{
        Set<String> keySet = hashMap.keySet();
        
        for (String str : keySet) 
        {
            word.set(str);
            one.set(hashMap.get(str));
            context.write(word, one);
        }
	}
}
